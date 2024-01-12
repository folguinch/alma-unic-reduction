"""Handler for the data."""
from typing import Optional, List, Dict, Tuple, Sequence
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, InitVar
from dataclasses import field as dcfield
from pathlib import Path
import json
import os

from casatasks import uvcontsub, concat, flagdata, flagmanager
from casatasks import split as split_ms
from casatools import synthesisutils
import astropy.units as u

from .utils import (get_spws_indices, validate_step, get_targets, get_array,
                    find_spws, extrema_ms, gaussian_beam,
                    gaussian_primary_beam, round_sigfig, continuum_bins,
                    flags_freqs_to_channels)
from .clean_tasks import get_tclean_params, tclean_parallel
from .common_types import SectionProxy

def new_array_dict(arrays: Sequence = ('12m', '7m')):
    return {array: () for array in arrays}

@dataclass
class FieldHandler:
    """Keep track of field data."""
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
    default_config: InitVar[Optional[Path]] = None
    """Default configuration file name."""
    datadir: InitVar[Path] = Path('./').resolve()
    """Data directory."""
    neb: Optional[int] = None
    """Number of execution blocks in the uvdata."""
    cont_file: Optional['pathlib.Path'] = None
    """Continuum file."""
    spws: List[str] = dcfield(default_factory=list, init=False)
    """List with spw values."""

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

    def _validate_config(self):
        """Check that data in config and FieldHandler coincide."""
        # Check if config has been initiated
        if self.config is None:
            raise ValueError('Cannot create FieldHandler without config')

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

    def update_spws(self):
        """Update the spws by reading the uvdata."""
        self.spws = get_spws_indices(self.uvdata)
        self.log.debug('Updated SPWs: %s', self.spws)

    def read_config(self,
                    default_config: Optional['pathlib.Path'] = None) -> None:
        """Load a configuration file."""
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
        """Update and write the configuration file."""
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

    def _flags_from_cont_file(self) -> Dict:
        """Read line flags from `cont_file`."""
        # Read file
        with self.cont_file.open() as cfile:
            lines = cfile.readlines()

        # Find source and get flags
        is_field = False
        flags = {}
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
                if spw not in flags:
                    flags[spw] = [line.split()[0]]
                else:
                    flags[spw].append(line.split()[0])
        
        # Copy flags for multi EBs
        final_flags = {}
        for i, spws in enumerate(self.spws):
            for spw in map(int, spws.split(',')):
                final_flags[spw] = flags[i]
        final_flags = dict(sorted(final_flags.items()))

        return final_flags

    def get_line_flags(self,
                       suffix: str,
                       invert: bool = False,
                       mask_borders: bool = False,
                       resume: bool = False) -> str:
        """Read `cont_file` and assign its lines to theirs `spws`."""
        flag_file = self.uvdata.with_suffix(f'{suffix}.json')
        if flag_file.is_file() and resume:
            flags = json.loads(flag_file.read_text())
        else:
            flags = self._flags_from_cont_file()
            self.log.debug('Flags (frequency): %s', flags)
            flags = flags_freqs_to_channels(flags, self.uvdata, invert=invert,
                                            mask_borders=mask_borders)
            flag_file.write_text(json.dumps(flags, indent=4))
        self.log.debug('Flags (channels): %s', flags)

        return flags

    def get_image_scales(self,
                         uvtype: str = '',
                         spw: Optional[int] = None,
                         beam_oversample: int = 5,
                         sigfig: int = 2,
                         pbcoverage: float = 1.) -> Tuple[str, int]:
        """Get cell and image sizes."""
        # Get parameters from MS
        uvdata = self.get_uvname(uvtype)
        low_freq, high_freq, longest_baseline = extrema_ms(uvdata, spw=spw)
        self.log.info('Frequency range: %s - %s', low_freq, high_freq)
        self.log.info('Longest baseline: %s', longest_baseline)

        # Size
        beam = gaussian_beam(high_freq, longest_baseline)
        pbsize = gaussian_primary_beam(low_freq, u.Quantity(self.array))
        self.log.info('Estimated smallest beam size: %s', beam)
        self.log.info('Estimated largest pb size: %s', pbsize)

        # Estimate cell size
        su = synthesisutils()
        cell = round_sigfig(beam / beam_oversample, sigfig=sigfig)
        self.log.info('Estimated cell size: %s', cell)
        imsize = int(2 * pbsize * pbcoverage / cell)
        self.log.info('Estimated image size: %s', imsize)
        imsize = su.getOptimumSize(imsize)
        self.log.info('Optimal image size: %s', imsize)

        return f'{cell.value}{cell.unit}', imsize

    def clean_data(self,
                   section: str,
                   imagename: Path,
                   uvtype: str = '',
                   nproc: int = 5,
                   resume: bool = False,
                   **tclean_args):
        """Run tclean on input or stored uvdata.

        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          section: configuration section.
          imagename: image directory path.
          uvtype: optional; type of uvdata to clean.
          nproc: optional; number of processors for parallel clean.
          resume: optional; resume where it was left?
          tclean_args: optional; additional arguments for tclean.
        """
        # tclean args
        config = self.config[section]
        kwargs = get_tclean_params(config, cfgvars=tclean_args)

        # Check for files
        if imagename.exists() and resume:
            return
        elif imagename.exists() and not resume:
            self.log.warning('Deleting image: %s', imagename)
            os.system(f"rm -rf {imagename.with_suffix('.*')}")

        # Run tclean
        uvdata = self.get_uvname(uvtype)
        tclean_parallel(uvdata, imagename.with_name(imagename.stem),
                        nproc, kwargs, log=self.log)

    def clean_per_spw(self,
                      section: str,
                      outdir: Path = Path('./'),
                      uvtype: str = '',
                      nproc: int = 5,
                      resume: bool = False,
                      **tclean_args):
        """Run `tclean` on all spws.
        
        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          section: configuration section.
          outdir: output directory path.
          uvtype: optional; type of uvdata to clean.
          nproc: optional; number of processors for parallel clean.
          resume: optional; resume where it was left?
          tclean_args: optional; additional arguments for tclean.
        """
        # Iterate over spws
        for i, spw in enumerate(self.spws):
            # Clean args
            suffix = self.get_uvsuffix(uvtype)
            imagename = outdir / (f'{self.name}_{self.array}_spw{i}'
                                  f'{suffix}.image')

            # Check for files
            if imagename.exists() and resume:
                self.log.info('Skipping cube for spw %i', i)
                continue
            elif imagename.exists() and not resume:
                self.log.warning('Deleting cubes for spw %i', i)
                os.system(f"rm -rf {imagename.with_suffix('.*')}")

            # Parameters
            cell = self.config.get(section, 'cell', vars=tclean_args,
                                   fallback=None)
            imsize = self.config.get(section, 'imsize', vars=tclean_args,
                                     fallback=None)
            impars = {}
            if cell is None or imsize is None:
                new_cell, new_imsize = self.get_image_scales(spw=i)
                if cell is None:
                    impars['cell'] = new_cell
                if imsize is None:
                    impars['imsize'] = f'{new_imsize}'

            # Run clean
            self.clean_data(section, imagename, uvtype=uvtype, nproc=nproc,
                            resume=resume, spw=spw, **tclean_args, **impars)
            print('-' * 80)

    def contsub(self,
                section: str = 'contsub',
                resume: bool = False) -> None:
        """Run `uvcontsub` to obtain continuum subtracted visibilities."""
        val_step = validate_step(resume, self.uvcontsub, log=self.log.warning)
        if not val_step:
            return

        # Prepare input
        suffix = self.get_uvsuffix('uvcontsub')
        mask_borders = self.config.getboolean(section, 'mask_borders')
        fitspec = self.get_line_flags(suffix, invert=True,
                                      mask_borders=mask_borders, resume=resume)
        fitorder = self.config.getint(section, 'fitorder')

        # Run uvcontsub
        uvcontsub(vis=f'{self.uvdata}', outputvis=f'{self.uvcontsub}',
                  fitspec=fitspec, fitorder=fitorder)

    def continuum_vis(self,
                      section: str = 'continuum',
                      resume: bool = False) -> None:
        """Generate continuum visibilities.

        It generates 2 visibility sets:

        - Average of all the channels using the input width.
        - Average of the line free channels using the `cont_file` and width.
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
        width = self.config.get(section, 'width', fallback=None)
        if width is not None:
            width = list(map(int, width.split(',')))
        else:
            width = continuum_bins(self.uvdata, u.Quantity(self.array))
            self.config[section]['width'] = ','.join(map(str, width))
            self.write_config()
        self.log.info('Continuum bins: %s', width)
        datacolumn = self.config[section]['datacolumn']

        if val_all:
            # Average channels without flagging
            split_ms(vis=f'{self.uvdata}', outputvis=f'{allchans}', width=width,
                     datacolumn=datacolumn)
        if val_lf:
            # Get line free channels
            suffix = self.get_uvsuffix('continuum')
            flag_chans = self.get_line_flags(suffix, resume=resume)

            # Split the uvdata
            # WARNING: this needs to be tested
            # otherwise, use flagmanager and then split the visibilities
            flagmanager(vis=f'{self.uvdata}', mode='save',
                        versionname='before_cont_flags')
            flagdata(vis=f'{self.uvdata}', mode='manual', spw=flag_chans,
                     flagbackup=False)
            split_ms(vis=f'{self.uvdata}', outputvis=f'{linefree}',
                     width=width, datacolumn=datacolumn)
            flagmanager(vis=f'{self.uvdata}', mode='restore',
                        versionname='before_cont_flags')

        return linefree, allchans

@dataclass
class FieldManager:
    """Manage field information."""
    log: 'logging.Logger'
    """Logging object."""
    resume: bool = False
    """Resume or delete already created data?"""
    original: Dict[str, Tuple[Path]] = dcfield(default_factory=new_array_dict)
    """Original data for each array."""
    split: Dict[str, Tuple[Path]] = dcfield(default_factory=new_array_dict)
    """Split data for each array."""
    data_handler: Dict[str, FieldHandler] = dcfield(default_factory=dict)
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
        if original_uvdata is not None:
            self.append_data(field=field,
                             original_uvdata=original_uvdata,
                             default_config=default_config,
                             configfile=configfile,
                             cont_file=cont_file,
                             datadir=datadir)

    @property
    def name(self):
        return self.data_handler[list(self.data_handler.keys())[0]].field

    def datadir(self, array: str) -> Path:
        return self.data_handler[array].uvdata.parent

    def _init_data_handler(self,
                           field: str,
                           array: str,
                           **kwargs):
        self.data_handler[array] = FieldHandler(field,
                                                array,
                                                self.log,
                                                **kwargs)

    def append_data(self,
                    field: Optional[str] = None,
                    original_uvdata: Optional[Path] = None,
                    #split_uvdata: Optional[Path] = None,
                    default_config: Optional[Path] = None,
                    configfile: Optional[Path] = None,
                    cont_file: Optional[Path] = None,
                    datadir: Optional[Path] = None):
        """Update information in field."""
        if original_uvdata is not None:
            # Store original data
            array = get_array(original_uvdata)
            self.log.debug('Data array: %s', array)
            self.original[array] += (original_uvdata,)
            self.log.debug('Original data: %s', self.original)

            # Update handler
            neb = len(self.original[array])
            if (handler:= self.data_handler.get(array)) is None:
                self.log.debug('Generating new data handler')
                self._init_data_handler(field,
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

            # Split?
            split = self.datadir(array) / f'{self.name}_{array}_eb{neb}.ms'
            if split.exists() and not self.resume:
                self.log.warning('Deleting split ms: %s', split)
                os.system(f'rm -rf {split}')
            if not split.exists():
                self.log.info('Splitting input ms to: %s', split)
                self.split_field(array, original_uvdata, split)
            else:
                self.log.info('Found split ms: %s', split)
            self.split[array] += (split,)
            self.log.debug('Split data: %s', self.split)

    def split_field(self, array: str, uvdata: Path, splitvis: Path):
        """Split field from input uvdata."""
        # Split parameters
        if ((spw := self.data_handler[array].config.get(
            'split', 'spw', fallback=None)) is None):
            spw = find_spws(self.name, uvdata)
            self.data_handler[array].update_config(split={'spw': spw})
        datacolumn = self.data_handler[array].config['split']['datacolumn']

        # Perform splitting
        self.log.info('Selected spws for split: %s', spw)
        self.log.info('Selected datacolumn for split: %s', datacolumn)
        split_ms(vis=f'{uvdata}', outputvis=f'{splitvis}', field=self.name,
                 datacolumn=datacolumn, spw=spw)

    def concat_data(self):
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

            # Concat
            if len(vis) == 0:
                self.log.warning('Data handler created but no data: %s', array)
                raise ValueError('Missing data')
            elif len(vis) == 1:
                self.log.info('Moving %s to: %s', vis[0], concatvis)
                os.system(f'mv {vis[0]} {concatvis}')
            else:
                self.log.info('MSs to concat: %s', vis)
                self.log.info('Concatenating MSs to: %s', concatvis)
                concat(vis=vis, concatvis=f'{concatvis}')

            # Update handler spws
            handler.update_spws()

    def dirty_cubes(self,
                    section: str = 'dirty_cubes',
                    uvtype: str = '',
                    nproc: int = 5,
                    arrays: Optional[Sequence[str]] = None,
                    outdir: Optional[Path] = None):
        """Calculate dirty cubes per spw."""
        # Select arrays
        if arrays is None:
            arrays = self.data_handler.keys()

        # Iterate over selected arrays
        for array in arrays:
            # Check input
            handler = self.data_handler[array]
            if outdir is None:
                outdir = handler.uvdata.parent / section
            outdir.mkdir(exist_ok=True)

            # Global imaging sizes
            cell = handler.config.get(section, 'cell', fallback=None)
            imsize = handler.config.get(section, 'imsize', fallback=None)
            if cell is None or imsize is None:
                new_cell, new_imsize = handler.get_image_scales()
                if cell is None:
                    handler.config[section]['cell'] = new_cell
                if imsize is None:
                    handler.config[section]['imsize'] = f'{new_imsize}'
                handler.write_config()

            # Clean spws
            self.log.info('Computing dirty cubes for array: %s', array)
            handler.clean_per_spw(section, outdir=outdir, nproc=nproc,
                                  uvtype=uvtype, resume=self.resume, niter=0)
            print('=' * 80)

    def contsub_visibilities(self,
                             dirty_images: bool = False,
                             nproc: int = 5):
        """Compute continuum subtracted visibilities."""
        for array, handler in self.data_handler.items():
            self.log.info('Computing contsub visibilities for array: %s', array)
            handler.contsub(resume=self.resume)

            # Compute dirty
            if dirty_images:
                self.dirty_cubes(uvtype='uvcontsub', nproc=nproc)

    def continuum_visibilities(self,
                               control_image: bool = False,
                               nproc: int = 5,
                               outdir: Optional[Path] = None):
        """Compute continuum visibilities for data."""
        for array, handler in self.data_handler.items():
            self.log.info('Computing cont visibilities for array: %s', array)
            handler.continuum_vis(resume=self.resume)

            # Make a control image
            if control_image:
                # Parameters
                tclean_args = {}
                section = 'continuum'
                uvtype = 'continuum'
                tclean_args['cell'] = handler.config.get(section, 'cell',
                                                         fallback=None)
                tclean_args['imsize'] = handler.config.get(section, 'imsize',
                                                           fallback=None)
                if tclean_args['cell'] is None or tclean_args['imsize'] is None:
                    impars = handler.get_image_scales(uvtype=uvtype)
                    if tclean_args['cell'] is None:
                        tclean_args['cell'] = impars[0]
                    if tclean_args['imsize'] is None:
                        tclean_args['imsize'] = f'{impars[1]}'
                if outdir is None:
                    outdir = handler.uvdata.parent / f'{section}_control'
                outdir.mkdir(exist_ok=True)
                suffix = handler.get_uvsuffix(uvtype)
                imagename = outdir / f'{handler.name}_{array}{suffix}.image'
                handler.clean_data(section, imagename, uvtype=uvtype,
                                   nproc=nproc, resume=self.resume,
                                   **tclean_args)

    def write_configs(self):
        """Write configs to disk."""
        for handler in self.data_handler.values():
            handler.write_config()

class DataManager(Dict):
    """Manage pipeline data."""
    def __init__(self, log: 'logging.Logger', **kwargs):
        self.log = log
        super().__init__(kwargs)

    @classmethod
    def from_uvdata(cls,
                    uvdata: Path,
                    default_config: Path,
                    log: 'logging.Logger',
                    datadir: Path = Path('./'),
                    cont: Optional[Sequence[Path]] = None,
                    resume: bool = False):
        """Generate a `DataManager` from input MSs.

        It uses the input uvdata to find target fields and generates a
        configuration file for each.
        """
        # First pass: iterate over uvdata and define fields
        fields = {}
        for i, ms in enumerate(uvdata):
            print('=' * 80)
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
                print('-' * 80)
                if field in fields:
                    log.info('Updating field: %s', field)
                    fields[field].append_data(original_uvdata=ms)
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
        # files
        for name, field in fields.items():
            print('=' * 80)
            log.info('Concatenating field: %s', name)
            field.concat_data()
            log.info('Writing configs')
            field.write_configs()

        return cls(log=log, **fields)

    @classmethod
    def from_configs(cls,
                     configfiles: Path,
                     log: 'logging.Logger',
                     datadir: Path = Path('./'),
                     cont: Optional[Sequence[Path]] = None,
                     resume: bool = False):
        """Generate a `DataManager` from input configuration files.

        At the moment it assumes the data is ready for processesing.
        """
        # Iterate over files
        fields = {}
        for i, cfg in enumerate(configfiles):
            # Create field handler
            if cont is not None:
                cont_file = cont[i]
            else:
                cont_file = None
            handler = FieldHandler(configfile=cfg, datadir=datadir,
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
