"""Handlers for managing and processing data."""
from typing import Optional, List, Dict, Tuple, Sequence
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, InitVar
from dataclasses import field as dcfield
from pathlib import Path
import json
import os

from casatasks import uvcontsub, concat, flagdata, flagmanager, exportfits
from casatasks import split as split_ms
from casatools import synthesisutils
import astropy.units as u

from .utils import (get_spws_indices, validate_step, get_targets, get_array,
                    find_spws, extrema_ms, gaussian_beam,
                    gaussian_primary_beam, round_sigfig, continuum_bins,
                    flags_from_cont_ranges, clumps_to_casa)
from .clean_tasks import (get_tclean_params, tclean_parallel,
                          recommended_auto_masking, auto_thresh_clean,
                          cube_multi_clean, cube_multi_images)
from .plotting import (plot_imaging_products, plot_comparison,
                       plot_imaging_spectra)
from .spectrum_tools import extract_spectrum, plot_spectral_selection

def new_array_dict(arrays: Sequence = ('12m', '7m')):
    return {array: () for array in arrays}

@dataclass
class ArrayHandler:
    """Keep track of array data."""
    name: Optional[str] = None
    """Name of the field."""
    array: Optional[str] = None
    """Array name."""
    config: Optional[ConfigParser] = dcfield(default=None, init=False)
    """Configuration parser."""
    log: 'logging.Logger' = None
    """Logging object."""
    uvdata: Optional[Path] = None
    """Measurement set directory."""
    configfile: Optional[Path] = None
    """Configuration file name."""
    neb: Optional[int] = None
    """Number of execution blocks in the uvdata."""
    cont_file: Optional['pathlib.Path'] = None
    """Continuum file."""
    spws: List[str] = dcfield(default_factory=list, init=False)
    """List with spw values."""
    spectra: List[Tuple[u.Quantity]] = dcfield(default_factory=list,
                                               init=False)
    """Stored spectra per SPW."""
    default_config: InitVar[Optional[Path]] = None
    """Default configuration file name."""
    datadir: InitVar[Path] = Path('./').resolve()
    """Data directory."""

    def __post_init__(self, default_config, datadir):
        # Check log
        if self.log is None:
            raise NotImplementedError('A logging.Logger is required')

        # Check uvdata name
        if (self.uvdata is None and
            self.field is not None and
            self.array is not None):
            self.uvdata = datadir / f'{self.field}_{self.array}.ms'
            self.log.debug('Setting uvdata name as: %s', self.uvdata)

        # Open configfile
        if self.configfile is not None or default_config is not None:
            if self.configfile is None:
                self.configfile = self.uvdata.with_suffix('.cfg')
            self.read_config(default_config=default_config)
        self._validate_config()

        # Fill spws
        if self.uvdata is not None and self.uvdata.exists():
            self.spws = get_spws_indices(self.uvdata)
            self.log.debug('SPWs in data: %s', self.spws)

    @property
    def field(self):
        return self.name

    @field.setter
    def field(self, value: str):
        self.name = value

    @property
    def uvcontinuum(self):
        return self.uvdata.with_suffix('.cont.ms')

    @property
    def uvcontsub(self):
        return self.uvdata.with_suffix('.contsub.ms')

    def get_antenna_diams(self, array: str) -> List[u.Quantity]:
        """Return the antenna diameters based on array name."""
        if array.count('m') > 1:
            arrays = array.split('m')[:-1]
            antennae = list(map(lambda x: float(x) * u.m, arrays))
        else:
            antennae = [u.Quantity(array)]

        return antennae

    def _validate_config(self):
        """Check that data in config and `ArrayHandler` coincide."""
        # Check if config has been initiated
        if self.config is None:
            raise ValueError('Cannot create ArrayHandler without config')

        # Field name
        if self.field is None:
            self.field = self.config['DEFAULT'].get('field')
            self.log.debug('Copying field from config: %s', self.field)
        elif self.field != self.config['DEFAULT']['field']:
            self.log.debug('Updating field in config to: %s', self.field)
            self.config['DEFAULT']['field'] = self.field

        # Check array
        if self.array is None:
            self.array = self.config['DEFAULT'].get('array')
            self.log.debug('Copying array from config: %s', self.array)
        elif self.array != self.config['DEFAULT']['array']:
            self.log.debug('Updating array in config to: %s', self.array)
            self.config['DEFAULT']['array'] = self.array

        # Check uvdata
        if self.uvdata is None:
            datadir = Path(self.config['DEFAULT']['basedir'])
            self.uvdata = datadir / f'{self.field}_{self.array}.ms'
            self.log.debug('Setting uvdata name as: %s', self.uvdata)
        else:
            basedir1 = self.uvdata.parent
            basedir2 = Path(self.config['DEFAULT']['basedir'])
            if basedir1 != basedir2:
                self.log.debug('Updating datadir in config to: %s', basedir1)
                self.config['DEFAULT']['basedir'] = f'{basedir1}'
            else:
                pass

        # Check number of EBs
        if self.neb is None:
            self.neb = self.config['DEFAULT'].getint('neb')
            self.log.debug('Copying number of EBs from config: %s', self.neb)
        elif self.neb != self.config['DEFAULT'].getint('neb'):
            self.log.debug('Updating number of EBs in config to: %s', self.neb)
            self.config['DEFAULT']['neb'] = f'{self.neb}'

        # Store continuum file
        if self.cont_file is not None:
            self.config['continuum']['cont_file'] = f'{self.cont_file}'
            self.log.debug('Setting config cont file: %s', self.cont_file)
        elif (self.cont_file is None and
              (cont_file:=self.config.get('continuum', 'cont_file',
                                          fallback=None)) is not None):
            self.cont_file = Path(cont_file)
            self.log.debug('Setting cont file: %s', self.cont_file)

    def update_spws(self, uvtype: str = ''):
        """Update the spws by reading the uvdata."""
        self.spws = get_spws_indices(self.get_uvname(uvtype))
        self.log.debug('Updated SPWs: %s', self.spws)

    def read_config(self,
                    default_config: Optional['pathlib.Path'] = None) -> None:
        """Load a configuration file.
        
        Args:
          default_config: Optional; Default configuration scheme.
        """
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        if default_config is not None and default_config.is_file():
            self.log.info(f'Reading default configuration: {default_config}')
            self.config.read(default_config)
        if self.configfile is not None and self.configfile.is_file():
            self.log.info(f'Reading input configuration: {self.configfile}')
            self.config.read(self.configfile)
        elif self.configfile is not None and not self.configfile.is_file():
            self.update_config(write=True,
                               DEFAULT={'field': self.field,
                                        'array': self.array,
                                        'basedir': f'{self.uvdata.parent}',
                                        'neb': f'{self.neb}'})

    def update_config(self, write: bool = False, **sections):
        """Update and write the configuration file.
        
        Args:
          write: Optional; Save to disk.
          sections: Sections to update with values.
        """
        # Update file
        self.config.read_dict(sections)

        # Write
        if write:
            self.write_config()

    def write_config(self):
        """Write config file."""
        # Write file
        self.log.info(f'Writing configuration: {self.configfile}')
        with self.configfile.open('w') as configfile:
            self.config.write(configfile)

    def get_uvname(self, uvtype: str):
        """Get uvdata name from `uvtype`."""
        if uvtype not in ['', 'continuum', 'uvcontsub']:
            raise KeyError(f'Unrecognized uv type: {uvtype}' )

        if uvtype == 'continuum':
            uvdata = self.uvcontinuum
        elif uvtype == 'uvcontsub':
            uvdata = self.uvcontsub
        else:
            uvdata = self.uvdata

        return uvdata

    def get_uvsuffix(self, uvtype: str):
        """Get a suffix for file names from `uvtype`."""
        if uvtype == 'continuum':
            suffix = '.cont'
        elif uvtype == 'uvcontsub':
            suffix = '.contsub'
        else:
            suffix = ''
        return suffix

    def get_imagename(self,
                      uvtype: str,
                      outdir: Optional[Path] = None,
                      section: Optional[str] = None,
                      **suffixes) -> Path:
        """Generate an image name for intent.
        
        Args:
          uvtype: Type of uv data.
          outdir: Optional; Output directory.
          section: Optional; Config section of the image.
          suffixes: Optional; Additional `key_value` for file name.
        """
        # Set and create output directory
        if outdir is None:
            if section is not None:
                outdir = self.uvdata.parent / section
                outdir.mkdir(exist_ok=True)
            else:
                outdir = Path('./')
        else:
            outdir.mkdir(exist_ok=True)

        # Obtain additional suffixes
        suffix = self.get_uvsuffix(uvtype)
        aux = '_'.join(f'{k}{v}' for k, v in suffixes.items())

        # Generate name path
        imagename = f'{self.name}_{self.array}'
        if len(aux) != 0:
            imagename = f'{imagename}_{aux}'
        imagename = outdir / f'{imagename}{suffix}.image'

        return imagename

    def _cont_from_file(self) -> Dict[int, List[str]]:
        """Read continuum ranges from `cont_file`."""
        # Read file
        with self.cont_file.open() as cfile:
            lines = cfile.readlines()

        # Find source and get continuum ranges
        is_field = False
        cont = {}
        spw_orig = list(map(int, self.config['split']['spw'].split(',')))
        for line in lines:
            if line.startswith('Field') and self.name in line:
                is_field = True
                continue
            elif line.startswith('Field') and is_field:
                break
            if is_field and 'SpectralWindow' in line:
                number = int(line.split()[1])
                spw = spw_orig.index(number)
            elif '~' not in line:
                continue
            elif is_field and '~' in line:
                if spw not in cont:
                    cont[spw] = [line.split()[0]]
                else:
                    cont[spw].append(line.split()[0])

        # Copy continuum ranges for multi EBs
        final_cont = {}
        for i, spws in enumerate(self.spws):
            for spw in map(int, spws.split(',')):
                final_cont[spw] = cont[i]
        final_cont = dict(sorted(final_cont.items()))

        return final_cont

    def get_spectral_ranges(self,
                            suffix: str,
                            invert: bool = False,
                            mask_borders: bool = False,
                            resume: bool = False) -> Dict[int,
                                                          List[Tuple[int]]]:
        """Get spectral ranges from continuum file.
        
        The `cont.dat` contains the spectral ranges where the continuum is
        calculated. The `invert` parameter allows to select the channels where
        lines are.

        Args:
          suffix: File name suffix.
          invert: Optional; Reverse channel selection?
          mask_borders: Optional; Mask 10 channels at beginning and end?
          reume: Optional; Resume where it was left last time?

        Returns:
          A dictionary mapping SPW numbers with channel ranges.
        """
        flag_file = self.uvdata.with_suffix(f'{suffix}.json')
        if flag_file.is_file() and resume:
            flags = json.loads(flag_file.read_text())
        else:
            flags = self._cont_from_file()
            self.log.debug('Flags (frequency): %s', flags)
            flags = flags_from_cont_ranges(flags, self.uvdata, invert=invert,
                                           mask_borders=mask_borders)
            flag_file.write_text(json.dumps(flags, indent=4))
        self.log.debug('Flags (channels): %s', flags)

        return flags

    def get_image_scales(self,
                         uvtype: str = '',
                         spw: Optional[int] = None,
                         beam_oversample: int = 5,
                         sigfig: int = 2,
                         pbcoverage: float = 2.,
                         section: str = 'imaging') -> Tuple[str, int]:
        """Get cell and image sizes.
        
        Args:
          uvtype: Optional; Type of uvdata to image.
          spw: Optional; Select spectral window.
          beam_oversample: Optional; Number of pixels per beam.
          sigfig: Optional; Significant figures of the pixel size.
          pbcoverage: Optional; Number of PB sizes for the image size.

        Returns:
          CASA compatible cell size and image size.
        """
        # Search in config
        cell = self.config.get(section, 'cell', fallback=None)
        imsize = self.config.getint(section, 'imsize', fallback=None)
        if cell is not None and imsize is not None:
            return cell, imsize

        # Get parameters from MS
        uvdata = self.get_uvname(uvtype)
        low_freq, high_freq, longest_baseline = extrema_ms(uvdata, spw=spw)
        self.log.info('Frequency range: %s - %s', low_freq, high_freq)
        self.log.info('Longest baseline: %s', longest_baseline)

        # Size
        antenna = min(self.get_antenna_diams(self.array))
        beam = gaussian_beam(high_freq, longest_baseline)
        pbsize = gaussian_primary_beam(low_freq, antenna)
        self.log.info('Estimated smallest beam size: %s', beam)
        self.log.info('Estimated largest pb size: %s', pbsize)

        # Estimate cell size
        su = synthesisutils()
        if cell is None:
            cell = round_sigfig(beam / beam_oversample, sigfig=sigfig)
            self.log.info('Estimated cell size: %s', cell)
            cell = f'{cell.value}{cell.unit}'
            self.config.read_dict({section: {'cell': cell}})
            self.write_config()
        if imsize is None:
            imsize = int(pbsize * pbcoverage / cell)
            self.log.info('Estimated image size: %s', imsize)
            imsize = su.getOptimumSize(imsize)
            self.log.info('Optimal image size: %s', imsize)
            self.config[section]['imsize'] = f'{imsize}'
            self.write_config()

        return cell, imsize

    def clean_data(self,
                   section: str,
                   uvtype: str,
                   imagename: Optional[Path] = None,
                   nproc: int = 5,
                   auto_threshold: bool = False,
                   export_fits: bool = False,
                   resume: bool = False,
                   threshold_opt: Optional[str] = None,
                   plot_results: bool = False,
                   **tclean_args) -> Path:
        """Run tclean on input or stored uvdata.

        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          section: Configuration section.
          uvtype: Type of uvdata to clean.
          imagename: Optional; Image directory path.
          nproc: Optional; Number of processors for parallel clean.
          auto_threshold: Optional; Calculate threshold from noise in initial
            dirty image?
          export_fits: Optional; Export image to FITS file?
          resume: Optional; Resume where it was left?
          threshold_opt: Optional; Threshold config option for resume.
          plot_results: Optional; plot image, residual and masks?
          tclean_args: Optional; Additional arguments for tclean.

        Returns:
          The image file name.
        """
        # tclean args
        config = self.config[section]
        kwargs = get_tclean_params(config, cfgvars=tclean_args)
        if uvtype == 'continuum':
            specmode = kwargs.setdefault('specmode', 'mfs')
            kwargs.setdefault('pbcor', True)
        elif uvtype in ['', 'uvcontsub']:
            specmode = kwargs.setdefault('specmode', 'cube')
            kwargs.setdefault('outframe', 'LSRK')
        else:
            raise ValueError(f'Type {uvtype} not recognized')
        deconvolver = kwargs.setdefault('deconvolver', 'hogbom')
        kwargs.setdefault('weighting', 'briggs')
        robust = kwargs.setdefault('robust', 0.5)
        kwargs.setdefault('gridder', 'standard')
        kwargs.setdefault('niter', 100000)

        # Recover threshold
        if threshold_opt is None:
            if 'threshold' in kwargs:
                threshold_opt = 'threshold'
            else:
                threshold_opt = f'threshold_robust{robust}'
        if threshold_opt in config:
            self.log.info('Recovering threshold from %s in config',
                          threshold_opt)
            kwargs['threshold'] = config[threshold_opt]

        # Set imagename
        if imagename is None:
            imagename = self.get_imagename(uvtype, section=section,
                                           robust=robust)

        # Masking
        if kwargs.get('usemask', '') == 'auto-multithresh':
            kwargs = recommended_auto_masking(self.array) | kwargs

        # Check for files
        do_clean = True
        if deconvolver == 'mtmfs':
            realimage = imagename.with_suffix('.image.tt0')
        elif (specmode == 'cube' and
              config.getboolean('use_multi_clean')):
            _, realimage = cube_multi_images(imagename)
            kwargs['usemask'] = 'user'
        else:
            realimage = imagename
        self.log.debug('Real image name: %s', realimage)
        if realimage.exists() and resume:
            self.log.info('Skipping imaging for: %s', realimage)
            do_clean = False
            kwargs['threshold'] = config.get(threshold_opt, fallback=None)
            self.log.info('Recovered threshold: %s', kwargs['threshold'])
            imagename = realimage.with_suffix('.image')
        elif realimage.exists() and not resume:
            self.log.warning('Deleting image and products: %s', realimage)
            if (kwargs['specmode'] == 'cube' and
                config.getboolean('use_multi_clean')):
                os.system(f"rm -rf {realimage.with_suffix('.*')}")
            else:
                os.system(f"rm -rf {imagename.with_suffix('.*')}")

        # Run tclean
        if do_clean:
            uvdata = self.get_uvname(uvtype)
            if auto_threshold and 'threshold' not in kwargs:
                nsigma = config.get('thresh_nsigma', fallback=None)
                if nsigma is not None:
                    nsigma = tuple(map(float, nsigma.split(',')))
                    self.log.info('Cleaning down to %s sigma', nsigma)
                elif 'nsigma' in kwargs:
                    self.log.info('Using tclean nsigma=%f', kwargs['nsigma'])
                if (specmode == 'cube' and
                    config.getboolean('use_multi_clean')):
                    self.log.info('Using cube_multi_clean')
                    imagename, kwargs['threshold'] = cube_multi_clean(
                        uvdata,
                        imagename,
                        nproc,
                        self.array,
                        kwargs,
                        nsigma=nsigma,
                        plot_results=plot_results,
                        resume=resume,
                        log=self.log,
                    )
                    kwargs['usemask'] = 'user'
                else:
                    self.log.info('Using auto_thresh_clean')
                    kwargs['threshold'] = auto_thresh_clean(
                        uvdata,
                        imagename,
                        nproc,
                        kwargs,
                        nsigma=nsigma,
                        log=self.log,
                    )
                if nsigma is not None:
                    self.log.info('Automatic final threshold: %s',
                                  kwargs['threshold'])
                    self.update_config(write=True,
                                       **{section:
                                          {threshold_opt: kwargs['threshold']}})
            elif 'threshold' in kwargs:
                self.log.info('Threshold for clean: %s', kwargs['threshold'])
                tclean_parallel(uvdata, imagename.with_name(imagename.stem),
                                nproc, kwargs, log=self.log)
            else:
                self.log.info('Cleaning without threshold')
                tclean_parallel(uvdata, imagename.with_name(imagename.stem),
                                nproc, kwargs, log=self.log)

        # Export fits?
        fitsimage = realimage.with_suffix(f'{realimage.suffix}.fits')
        if export_fits and (do_clean or not fitsimage.exists()):
            self.log.info('Exporting image to FITS')
            exportfits(imagename=f'{realimage}', fitsimage=f'{fitsimage}',
                       overwrite=True)
            if kwargs.get('pbcor'):
                pbcorimage = realimage.with_suffix(f'{realimage.suffix}.pbcor')
                pbcorfits = pbcorimage.with_suffix('.pbcor.fits')
                exportfits(imagename=f'{pbcorimage}', fitsimage=f'{pbcorfits}',
                           overwrite=True)


        # Plot images
        if plot_results and imagename.exists():
            outdir = self.uvdata.parent / 'plots'
            outdir.mkdir(exist_ok=True)
            masking = kwargs.get('usemask')
            if specmode == 'cube':
                self.log.info('Plotting spectrum')
                plotname = imagename.with_suffix(f'.{section}.spectrum.png')
                plotname = outdir / plotname.name
                chans = plot_imaging_spectra(imagename, plotname,
                                             threshold=kwargs.get('threshold'),
                                             masking=masking,
                                             resume=resume)
            else:
                chans = None
            self.log.info('Plotting imaging results')
            plotname = imagename.with_suffix(f'.{section}.results.png')
            plotname = outdir / plotname.name
            plot_imaging_products(imagename, plotname,
                                  kwargs['deconvolver'],
                                  masking,
                                  threshold=kwargs.get('threshold'),
                                  chans=chans,
                                  resume=resume)

        return realimage

    def clean_per_spw(self,
                      section: str,
                      uvtype: str,
                      outdir: Optional[Path] = None,
                      nproc: int = 5,
                      resume: bool = False,
                      auto_threshold: bool = False,
                      export_fits: bool = False,
                      get_spectra: bool = False,
                      plot_results: bool = False,
                      **tclean_args) -> List[Path]:
        """Run `tclean` on all spws.
        
        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          section: Configuration section.
          uvtype: Type of uvdata to clean.
          outdir: Optional; Output directory path.
          nproc: Optional; Number of processors for parallel clean.
          resume: Optional; Resume where it was left?
          get_spectra: Optional; Extract spectra from cubes per spw?
          auto_threshold: Optional; Calculate threshold from noise in initial
            dirty image?
          export_fits: Optional; Export image to FITS file?
          plot_results: Optional; Plot spectra, image, residual and masks?
          tclean_args: Optional; Additional arguments for tclean.

        Returns:
          A list with the image file names.
        """
        # Check spws
        if len(self.spws) == 0:
            self.log.warning('Missing SPWs, will read from current uvtype')
            self.update_spws(uvtype)

        # Specify spws to clean
        if spws_to_clean := self.config.get(section, 'spws_to_clean',
                                            fallback=None):
            spws_to_clean = tuple(map(lambda x: int(x.strip()),
                                      spws_to_clean.split(',')))

        # Iterate over spws
        imagenames = []
        for i, spw in enumerate(self.spws):
            self.log.info('.' * 80)
            if spws_to_clean is not None and i not in spws_to_clean:
                self.lof.info('Skipping SPW: %i (%s)', i, spw)
            self.log.info('Cleaning SPW: %i (%s)', i, spw)
            # Clean args
            robust = self.config.get(section, 'robust',
                                     vars=tclean_args, fallback=0.5)
            imagename = self.get_imagename(uvtype, outdir=outdir,
                                           section=section, spw=f'{i}',
                                           robust=robust)
            threshold_opt = f'threshold_spw{i}_robust{robust}'

            # Check for files
            #trigger = True
            #if imagename.exists() and resume:
            #    self.log.info('Skipping cube for spw %i', i)
            #    trigger = False
            #elif imagename.exists() and not resume:
            #    self.log.warning('Deleting cubes for spw %i', i)
            #    os.system(f"rm -rf {imagename.with_suffix('.*')}")

            # Run clean
            #if trigger:
            imagename = self.clean_data(section, uvtype,
                                        imagename=imagename,
                                        nproc=nproc,
                                        auto_threshold=auto_threshold,
                                        export_fits=export_fits,
                                        resume=resume,
                                        threshold_opt=threshold_opt,
                                        plot_results=plot_results,
                                        spw=spw,
                                        **tclean_args)
            imagenames.append(imagename)

            # Extract spectrum
            if get_spectra:
                self.log.info('Extracting spectra')
                fitsimage = imagename.with_suffix('.image.fits')
                if not fitsimage.exists():
                    exportfits(imagename=f'{imagename}',
                               fitsimage=f'{fitsimage}')
                self.spectra.append(extract_spectrum(fitsimage)[:-1])

        return imagenames

    def contsub(self,
                section: str = 'contsub',
                resume: bool = False,
                plot_selection: bool = False) -> None:
        """Run `uvcontsub` to obtain continuum subtracted visibilities.
        
        Args:
          section: Optional; Configuration section.
          resume: Optional; Continue from where it was left?
          plot_selection: Optional; Plot selection for `fitspec`?
        """
        val_step = validate_step(resume, self.uvcontsub, log=self.log.warning)
        if not val_step:
            return

        # Prepare input
        suffix = self.get_uvsuffix('uvcontsub')
        mask_borders = self.config.getboolean(section, 'mask_borders')
        fitspec = self.get_spectral_ranges(suffix, invert=True,
                                           mask_borders=mask_borders,
                                           resume=resume)
        fitorder = self.config.getint(section, 'fitorder', fallback=1)

        # Plot?
        if plot_selection and len(self.spectra) != 0:
            self.log.info('Plotting spectra for contsub')
            outdir = self.uvcontsub.parent / 'plots'
            outdir.mkdir(exist_ok=True)
            plot_spectral_selection(self.spectra, fitspec, self.uvdata,
                                    self.spws, outdir, suffix=suffix)
        elif plot_selection:
            self.log.info('No spectra to plot')

        # Run uvcontsub
        uvcontsub(vis=f'{self.uvdata}', outputvis=f'{self.uvcontsub}',
                  fitspec=clumps_to_casa(fitspec), fitorder=fitorder)

    def continuum_vis(self,
                      section: str = 'continuum',
                      resume: bool = False,
                      plot_selection: bool = False) -> Tuple[Path]:
        """Generate continuum visibilities.

        It generates 2 visibility sets:

        - Average of all the channels using the input width.
        - Average of the line free channels using the `cont_file` and width.

        Args:
          section: Optional; Configuration section.
          resume: Optional; Continue from where it was left?
          plot_selection: Optional; Plot selection for flagged channels?

        Returns:
          The line-free and averaged MS file names.
        """
        # Check cont_file
        if self.cont_file is None:
            self.log.warning('Continuum file not assigned, nothing to do')
            return

        # Check files
        allchans = self.uvdata.with_suffix('.cont_all.ms')
        linefree = self.uvcontinuum
        val_all = validate_step(resume, allchans, log=self.log.warning)
        val_lf = validate_step(resume, linefree, log=self.log.warning)

        # Values from config
        width = self.config.get(section, 'split_width', fallback=None)
        if width is not None:
            width = list(map(int, width.split(',')))
        else:
            width = continuum_bins(self.uvdata, u.Quantity(self.array),
                                   log=self.log.info)
            self.config[section]['split_width'] = ','.join(map(str, width))
            self.write_config()
        self.log.info('Continuum bins: %s', width)
        datacolumn = self.config.get(section, 'datacolumn', fallback='data')

        # Split continuum unflagged
        if val_all:
            # Average channels without flagging
            self.log.info('Calculating unflagged binned continuum')
            split_ms(vis=f'{self.uvdata}', outputvis=f'{allchans}', width=width,
                     datacolumn=datacolumn)
            self.log.info('-' * 80)

        # Split continuum
        if val_lf:
            # Get line free channels
            suffix = self.get_uvsuffix('continuum')
            mask_borders = self.config.getboolean(section, 'mask_borders')
            self.log.info('Obtaining line flags')
            flag_chans = self.get_spectral_ranges(suffix,
                                                  mask_borders=mask_borders,
                                                  resume=resume)
            casa_flag_chans = clumps_to_casa(flag_chans)

            # Plot?
            if plot_selection and len(self.spectra) != 0:
                self.log.info('Plotting spectra for line flags')
                outdir = linefree.parent / 'plots'
                outdir.mkdir(exist_ok=True)
                plot_spectral_selection(self.spectra, flag_chans, self.uvdata,
                                        self.spws, outdir, suffix=suffix)
            elif plot_selection:
                self.log.info('No spectra to plot')

            # Split the uvdata
            self.log.info('Flagging data')
            flagmanager(vis=f'{self.uvdata}', mode='save',
                        versionname='before_cont_flags')
            flagdata(vis=f'{self.uvdata}', mode='manual', spw=casa_flag_chans,
                     flagbackup=False)
            self.log.info('Splitting binned continuum')
            split_ms(vis=f'{self.uvdata}', outputvis=f'{linefree}',
                     width=width, datacolumn=datacolumn)
            self.log.info('Restoring flags')
            flagmanager(vis=f'{self.uvdata}', mode='restore',
                        versionname='before_cont_flags')

        return linefree, allchans

@dataclass
class FieldManager:
    """Manage field information for different array configurations."""
    log: 'logging.Logger'
    """Logging object."""
    resume: bool = False
    """Resume or delete already created data?"""
    original: Dict[str, Tuple[Path]] = dcfield(default_factory=new_array_dict)
    """Original data for each array."""
    split: Dict[str, Tuple[Path]] = dcfield(default_factory=new_array_dict)
    """Split data for each array."""
    data_handler: Dict[str, ArrayHandler] = dcfield(default_factory=dict)
    """Data handler."""
    field: InitVar[Optional[str]] = None
    """Field name."""
    original_uvdata: InitVar[Optional[Path]] = None
    """Original data."""
    datadir: InitVar[Path] = Path('./')
    """Data directory."""
    default_config: InitVar[Optional[Path]] = None
    """Default configuration file name."""
    configfile: InitVar[Optional[Path]] = None
    """Configuration file name."""
    cont_file: InitVar[Optional[Path]] = None
    """Continuum file."""

    def __post_init__(self, field, original_uvdata, datadir, default_config,
                      configfile, cont_file):
        # When original_data is given, then split the field and fill the object
        if original_uvdata is not None:
            self.log.info('Appending data from original MS')
            self.append_data(original_uvdata,
                             field=field,
                             default_config=default_config,
                             configfile=configfile,
                             cont_file=cont_file,
                             datadir=datadir)

    @property
    def name(self):
        """Name associated to the field."""
        return self.data_handler[list(self.data_handler.keys())[0]].field

    @property
    def arrays(self) -> Tuple[str]:
        """Names of stored arrays."""
        return tuple(self.data_handler.keys())

    def datadir(self, array: str) -> Path:
        """Data directory where uvdata is stored."""
        return self.data_handler[array].uvdata.parent

    def _init_data_handler(self,
                           field: str,
                           array: str,
                           **kwargs) -> ArrayHandler:
        # Shortcut
        self.data_handler[array] = ArrayHandler(field,
                                                array,
                                                self.log,
                                                **kwargs)

        return self.data_handler[array]

    def append_data(self,
                    original_uvdata: Path,
                    field: Optional[str] = None,
                    default_config: Optional[Path] = None,
                    configfile: Optional[Path] = None,
                    cont_file: Optional[Path] = None,
                    datadir: Optional[Path] = None):
        """Append or update data from original calibrated MS.
        
        Args:
          original_uvdata: Original uv data.
          field: Optional; Field name.
          default_config: Optional; Default configuration file name.
          configfile: Optional; Configuration file name.
          cont_file: Optional; Continuum file name.
          datadir: Optional; Data directory.
        """
        # Store original data for each array and calculate the number of EBs
        array = get_array(original_uvdata)
        self.log.info('Data array: %s', array)
        self.original[array] += (original_uvdata,)
        self.log.debug('Original data: %s', self.original)
        neb = len(self.original[array])

        # Update or create new ArrayHandler
        if (handler:= self.data_handler.get(array)) is None:
            self.log.debug('Generating new data handler')
            handler = self._init_data_handler(field,
                                              array,
                                              default_config=default_config,
                                              configfile=configfile,
                                              cont_file=cont_file,
                                              datadir=datadir,
                                              neb=neb)
        else:
            self.log.debug('Updating data handler')
            handler.neb = neb
            handler.update_config(DEFAULT={'neb': f'{neb}'})


        # Split the data from original MS?
        split = self.datadir(array) / f'{self.name}_{array}_eb{neb}.ms'
        uvdata = handler.uvdata
        if split.exists() and not self.resume:
            self.log.warning('Deleting split ms: %s', split)
            os.system(f'rm -rf {split}')
        if uvdata.exists() and not self.resume:
            self.log.warning('Deleting concat ms: %s', uvdata)
            os.system(f'rm -rf {uvdata}')
        if not split.exists() and not uvdata.exists():
            self.log.info('Splitting input ms to: %s', split)
            self.split_field(array, original_uvdata, split)
        elif uvdata.exists():
            self.log.info('Will not split original, uv data exists')
        else:
            self.log.info('Found split ms: %s', split)
        self.split[array] += (split,)
        self.log.debug('Split data: %s', self.split)

    def get_uvnames(self,
                    arrays: Sequence[str],
                    uvtype: str = '') -> Tuple[Path]:
        """Obtain uv data file names for an array.

        Args:
          arrays: Arrays to use.
          uvtype: Type of uv data (`'', 'continuum', 'uvcontsub'`).
        """
        uvnames = tuple()
        for array in arrays:
            uvname = self.data_handler[array].get_uvname(uvtype)
            if not uvname.exists():
                continue
            else:
                uvnames += (uvname,)

        return uvnames

    def split_field(self, array: str, uvdata: Path, splitvis: Path):
        """Split field from input uvdata.
        
        Args:
          array: Name of the array.
          uvdata: MS file name.
          splitvis: Output MS file name.
        """
        # Split parameters
        if ((spw := self.data_handler[array].config.get(
            'split', 'spw', fallback=None)) is None):
            spw = find_spws(uvdata)
            self.data_handler[array].update_config(split={'spw': spw})
        datacolumn = self.data_handler[array].config['split']['datacolumn']

        # Perform splitting
        self.log.info('Selected spws for split: %s', spw)
        self.log.info('Selected datacolumn for split: %s', datacolumn)
        split_ms(vis=f'{uvdata}', outputvis=f'{splitvis}', field=self.name,
                 datacolumn=datacolumn, spw=spw)

    def concat_data(self, cleanup: bool = False):
        """Concatenate stored data per array."""
        for array, handler in self.data_handler.items():
            # Prepare concat input
            vis = [f'{val}' for val in self.split[array]]
            concatvis = handler.uvdata

            # Check data in disk
            if concatvis.exists() and not self.resume:
                self.log.warning('Deleting concat ms: %s', concatvis)
                os.system(f'rm -rf {concatvis}')
            elif concatvis.exists() and self.resume:
                continue

            # Concat data if needed
            if len(vis) == 0:
                self.log.error('Data handler created but no data: %s', array)
                raise ValueError('Missing data')
            elif len(vis) == 1:
                self.log.info('Moving %s to: %s', vis[0], concatvis)
                os.system(f'mv {vis[0]} {concatvis}')
            else:
                self.log.info('MSs to concat: %s', vis)
                self.log.info('Concatenating MSs to: %s', concatvis)
                concat(vis=vis, concatvis=f'{concatvis}')
                if cleanup:
                    os.system('rm -rf ' + ' '.join(vis))

            # Update handler spws
            handler.update_spws()

    def array_imaging(self,
                      section: str,
                      uvtype: str,
                      arrays: Optional[Sequence[str]] = None,
                      outdir: Optional[Path] = None,
                      nproc: int = 5,
                      auto_threshold: bool = False,
                      per_spw: bool = False,
                      get_spectra: bool = False,
                      export_fits: bool = False,
                      plot_results: bool = False,
                      compare_to: Optional[str] = None,
                      **tclean_args) -> List[Path]:
        """Image an array.

        Args:
          section: Configuration section with `tclean` parameters.
          uvtype: Type of uv data to clean.
          arrays: Optional; Array to image.
          outdir: Optional; Output directory.
          nproc: Optional; Number of parallel processes for cleaning.
          auto_threshold: Optional; Calculate threshold from noise in initial
            dirty image?
          per_spw: Optional; Clean each SPW individually?
          get_spectra: Optional; Extract spectra from cubes per SPW?
          export_fits: Optional; Export image to FITS file?
          plot_results: Optional; Plot image, residual and masks?
          compare_to: Optional; Compare image with this execution?
          tclean_args: Optional; Additional `casatasks.tclean` arguments.
        """
        # Select arrays
        if arrays is None:
            arrays = self.data_handler.keys()

        # Iterate over selected arrays
        imagenames = []
        for array in arrays:
            # Environment variables
            handler = self.data_handler[array]
            # Skip 7m12m for uvtype=''
            if array == '7m12m' and not handler.get_uvname(uvtype).exists():
                self.log.info('Skipping imaging for %s', array)
                continue
            self.log.info('-' * 80)
            self.log.info('Cleaning array: %s', array)

            # Fill tclean parameters
            impars = handler.get_image_scales(uvtype=uvtype)
            tclean_args['cell'] = impars[0]
            tclean_args['imsize'] = impars[1]

            # Clean cases
            if per_spw:
                self.log.info('Cleaning per SPW')
                aux = handler.clean_per_spw(section,
                                            uvtype,
                                            outdir=outdir,
                                            nproc=nproc,
                                            auto_threshold=auto_threshold,
                                            spws_to_clean=spws_to_clean,
                                            resume=self.resume,
                                            export_fits=export_fits,
                                            get_spectra=get_spectra,
                                            plot_results=plot_results,
                                            **tclean_args)
                imagenames += aux
            else:
                self.log.info('Cleaning %s', uvtype)
                imagename = handler.clean_data(section,
                                               uvtype,
                                               nproc=nproc,
                                               auto_threshold=auto_threshold,
                                               resume=self.resume,
                                               export_fits=export_fits,
                                               plot_results=plot_results,
                                               **tclean_args)
                imagenames.append(imagename)

                if compare_to is not None:
                    self.log.info('Plotting comparison between %s and %s',
                                  section, compare_to)
                    if (robust:=tclean_args.get('robust')) is not None:
                        comp_image = handler.get_imagename(uvtype,
                                                           section=compare_to,
                                                           robust=robust)
                    else:
                        raise ValueError('Robust parameter needed for name')
                    plotname = imagename.with_suffix((f'.{section}'
                                                      f'.compare_{compare_to}'
                                                      '.png'))
                    plotname = handler.uvdata.parent / 'plots' / plotname.name
                    plot_comparison(comp_image, imagename, plotname)

        return imagenames

    def contsub_visibilities(self,
                             dirty_images: bool = False,
                             nproc: int = 5,
                             get_spectra: bool = False):
        """Compute continuum subtracted visibilities.
        
        Args:
          dirty_images: Optional; Compute dirty images?
          nproc: Optional; Number of parallel processes for cleaning.
          get_spectra: Optional; Extract spectra from dirty cubes?
        """
        for array, handler in self.data_handler.items():
            self.log.info('Computing contsub visibilities for array: %s', array)
            handler.contsub(resume=self.resume, plot_selection=True)
            self.log.info('-' * 80)

        # Compute dirty
        if dirty_images:
            self.log.info('Producing dirty images for contsub')
            imagenames = self.array_imaging('dirty_cubes', 'uvcontsub',
                                            nproc=nproc, per_spw=True,
                                            get_spectra=get_spectra, niter=0)
        self.log.info('=' * 80)

    def continuum_visibilities(self,
                               control_image: bool = False,
                               nproc: int = 5,
                               outdir: Optional[Path] = None):
        """Compute continuum visibilities for data.
        
        Args:
          control_image: Optional; Calculate control image?
          nproc: Optional; Number of parallel processes for cleaning.
          outdir: Optional; Output directory.
        """
        for array, handler in self.data_handler.items():
            self.log.info('Computing cont visibilities for array: %s', array)
            handler.continuum_vis(resume=self.resume, plot_selection=True)
            self.log.info('-' * 80)
        self.log.info('=' * 80)

        # Make a control image
        if control_image:
            self.log.info('Producing control images for continuum')
            self.array_imaging('continuum_control', 'continuum', outdir=outdir,
                               nproc=nproc, auto_threshold=True,
                               export_fits=True, plot_results=True)

    def combine_arrays(self,
                       arrays: Sequence[str],
                       uvtypes: Sequence[str] = ('uvcontsub', 'continuum'),
                       default_config: Optional[Path] = None,
                       datadir: Optional[Path] = None,
                       control_images: bool = False,
                       dirty_images: bool = False,
                       nproc: int = 5):
        """Combine uv data of different arrays."""
        # Useful values
        new_array = ''.join(arrays)
        neb = sum(self.data_handler[array].neb for array in arrays)

        # Create handler
        self._init_data_handler(self.name, new_array, datadir=datadir,
                                default_config=default_config, neb=neb)
        handler = self.data_handler[new_array]

        # Concatenate products
        for uvtype in uvtypes:
            # Select data to concatenate
            to_concat = list(map(str, self.get_uvnames(arrays, uvtype=uvtype)))
            concatvis = handler.get_uvname(uvtype)

            # Check there is data to concat
            if len(to_concat) <= 0:
                self.log.warning('Nothing to concat for uvtype %s', uvtype)
                continue
            self.log.info('Concatenating MSs for uvtype: %s', uvtype)

            # Concatenate
            if concatvis.exists() and not self.resume:
                self.log.warning('Deleting combined array data: %s', concatvis)
                os.system(f'rm -rf {concatvis}')
            if not concatvis.exists():
                self.log.debug('Concatenating MSs: %s', to_concat)
                concat(vis=to_concat, concatvis=f'{concatvis}')
            if uvtype == 'uvcontsub':
                handler.update_spws(uvtype=uvtype)

            # Control images
            if control_images:
                if uvtype == 'uvcontsub' and dirty_images:
                    self.array_imaging('dirty_cubes',
                                       uvtype,
                                       arrays=[new_array],
                                       per_spw=True,
                                       nproc=nproc,
                                       niter=0)
                elif uvtype == 'continuum':
                    self.array_imaging('continuum_control',
                                       uvtype,
                                       arrays=[new_array],
                                       auto_threshold=True,
                                       nproc=nproc,
                                       export_fits=True)

    def write_configs(self):
        """Write configs to disk."""
        for handler in self.data_handler.values():
            handler.write_config()

class DataManager(Dict):
    """Manage pipeline data.
    
    It associates each field with a `FieldManager`.
    """
    def __init__(self, log: 'logging.Logger', **kwargs):
        self.log = log
        super().__init__(kwargs)

    @classmethod
    def from_uvdata(cls,
                    uvdata: Sequence[Path],
                    default_config: Path,
                    log: 'logging.Logger',
                    datadir: Path = Path('./'),
                    cont: Optional[Sequence[Path]] = None,
                    cleanup: bool = False,
                    resume: bool = False):
        """Generate a `DataManager` from input MSs.

        It uses the input uvdata to find target fields and generates a
        configuration file for each.

        Args:
          uvdata: MS file name(s).
          default_config: A default configuration skeleton.
          log: A `logging` object.
          datadir: Optional; Where the data generated will be stored.
          cont: Optional; A list of continuum files for each `uvdata`.
          cleanup: Optional; Delete intermediate eb data.
          resume: Optional; Continue from where it was left?
        """
        # First pass: iterate over uvdata and define fields
        fields = {}
        for i, ms in enumerate(uvdata):
            log.info('=' * 80)
            log.info('Operating over MS: %s', ms)
            targets = get_targets(ms)
            log.info('Fields in MS: %s', targets)

            # Continuum file
            if cont is not None:
                if len(cont) == 1:
                    cont_file = cont[0]
                else:
                    cont_file = cont[i]
            else:
                cont_file = None

            # Iterate over fields
            for field in targets:
                log.info('-' * 80)
                if field in fields:
                    log.info('Updating field: %s', field)
                    fields[field].append_data(ms,
                                              field=field,
                                              default_config=default_config,
                                              cont_file=cont_file,
                                              datadir=datadir)
                    continue
                # Read config
                log.info('Setting up field: %s', field)
                field_info = FieldManager(log, resume=resume, field=field,
                                          original_uvdata=ms,
                                          default_config=default_config,
                                          cont_file=cont_file,
                                          datadir=datadir)
                fields[field] = field_info

        # Second pass: iterate over fields to concatenate data and save config
        # files.
        for name, field in fields.items():
            log.info('=' * 80)
            log.info('Concatenating field: %s', name)
            field.concat_data(cleanup=cleanup)
            log.info('Writing configs')
            field.write_configs()

        return cls(log=log, **fields)

    @classmethod
    def from_configs(cls,
                     configfiles: Sequence[Path],
                     log: 'logging.Logger',
                     datadir: Path = Path('./'),
                     cont: Optional[Sequence[Path]] = None,
                     resume: bool = False):
        """Generate a `DataManager` from input configuration files.

        At the moment it assumes the data is ready for processesing.

        Args:
          configfiles: A list of configuration files.
          log: A `logging` object.
          datadir: Optional; Where the data generated will be stored.
          cont: Optional; A list of continuum files for each `uvdata`.
          resume: Optional; Continue from where it was left?
        """
        # Iterate over files
        fields = {}
        for i, cfg in enumerate(configfiles):
            # Create field handler
            if cont is not None:
                cont_file = cont[i]
            else:
                cont_file = None
            handler = ArrayHandler(configfile=cfg, datadir=datadir,
                                   cont_file=cont_file, log=log)

            # Update field
            if (field := handler.field) in fields:
                if (array := handler.array) in fields[field].data_handler:
                    raise KeyError(
                        'Loading 2 configs for the same field and array')
                fields[field].data_handler[array] = handler
            else:
                fields[field] = FieldManager(
                    data_handler={handler.array: handler},
                    log=log,
                    resume=resume)

        return cls(log=log, **fields)
