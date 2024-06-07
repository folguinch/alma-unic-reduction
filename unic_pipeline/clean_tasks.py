"""Tools for running the `tclean` CASA task."""
from typing import Sequence, Optional, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess

from astropy.io import fits
from astropy.stats import mad_std
from casatasks import tclean, exportfits
import astropy.units as u

from .common_types import SectionProxy
from .plotting import plot_imaging_products, plot_imaging_spectra
from .utils import get_func_params, round_sigfig

def recommended_auto_masking(array: str) -> Dict:
    """Recommended auto-masking values per array.

    From the table in [CASA
    guide](https://casaguides.nrao.edu/index.php/Automasking_Guide).
    """
    params = {'sidelobethreshold': 2.0,
              'noisethreshold': 4.25,
              'minbeamfrac': 0.3,
              'lownoisethreshold': 1.5,
              'negativethreshold': 0.0}
    if array == '7m':
        params['sidelobethreshold'] = 1.25
        params['noisethreshold'] = 5.0
        params['minbeamfrac'] = 0.1
        params['lownoisethreshold'] = 2.0

    return params

def get_tclean_params(
    config: SectionProxy,
    required_keys: Sequence[str] = ('cell', 'imsize'),
    ignore_keys: Sequence[str] = ('vis', 'imagename'),
    float_keys: Sequence[str]  = ('robust', 'pblimit', 'pbmask',
                                  'sidelobethreshold', 'noisethreshold',
                                  'minbeamfrac', 'lownoisethreshold',
                                  'negativethreshold'),
    int_keys: Sequence[str] = ('niter', 'chanchunks', 'nterms'),
    bool_keys: Sequence[str] = ('interactive', 'parallel', 'pbcor',
                                'perchanweightdensity'),
    int_list_keys: Sequence[str] = ('imsize', 'scales'),
    cfgvars: Optional[Dict] = None,
) -> Dict:
    """Filter input parameters and convert values to the correct type.

    Args:
      config: `ConfigParser` section proxy with input parameters to filter.
      ignore_keys: Optional; `tclean` parameters to ignore.
      float_keys: Optional; `tclean` parameters to convert to float.
      int_keys: Optional; `tclean` parameters to convert to int.
      bool_keys: Optional; `tclean` parameters to convert to bool.
      int_list_keys: Optional; `tclean` parameters as list of integers.
      cfgvars: Optional; Parameters to replace those in config file.
    """
    # Get params
    tclean_pars = get_func_params(tclean, config, required_keys=required_keys,
                                  ignore_keys=ignore_keys,
                                  float_keys=float_keys, int_keys=int_keys,
                                  bool_keys=bool_keys,
                                  int_list_keys=int_list_keys,
                                  cfgvars=cfgvars)
    if len(tclean_pars['imsize']) == 1:
        tclean_pars['imsize'] = tclean_pars['imsize'] * 2

    return tclean_pars

def tclean_parallel(vis: Path,
                    imagename: Path,
                    nproc: int,
                    tclean_args: dict,
                    log: Optional['logging.Logger'] = None):
    """Run `tclean` in parallel.

    If the number of processes (`nproc`) is 1, then it is run in a single
    processor. The environmental variable `MPICASA` is used to run the code,
    otherwise it will use the `mpicasa` and `casa` available in the system.

    A new logging file is created by `mpicasa`. This is located in the same
    directory where the program is executed.

    Args:
      vis: Measurement set.
      imagename: Image file name.
      nproc: Number of processes.
      tclean_args: Other arguments for tclean.
      log: Optional; Logging function.
    """
    if nproc == 1:
        tclean_args.update({'parallel': False})
        if log is not None:
            log.debug('Parameters for tclean: %s', tclean_args)
        tclean(vis=str(vis), imagename=str(imagename), **tclean_args)
    else:
        # Save tclean params
        tclean_args.update({'parallel': True})
        paramsfile = imagename.parent / 'tclean_params.json'
        paramsfile.write_text(json.dumps(tclean_args, indent=4))
        if log is not None:
            log.debug('Parameters for tclean: %s', tclean_args)
            log.debug('Writing parameters in: %s', paramsfile)

        # Run
        cmd = os.environ.get('MPICASA', 'mpicasa -n {0} casa').format(nproc)
        script = Path(__file__).parent / 'run_tclean_parallel.py'
        logfile = datetime.now().isoformat(timespec='milliseconds')
        logfile = f'tclean_parallel_{logfile}.log'
        cmd = (f'{cmd} --nogui --logfile {logfile} '
               f'-c {script} {vis} {imagename} {paramsfile}')
        if log is not None:
            log.info('Running: %s', cmd)
        # pylint: disable=R1732
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        proc.wait()

def auto_thresh_clean(vis: Path,
                      imagename: Path,
                      nproc: int,
                      tclean_args: dict,
                      thresh_niter: int = 2,
                      nsigma: Tuple[float] = (3.,),
                      sigfig: int = 3,
                      log: Optional['logging.Logger'] = None) -> str:
    """Find a threshold value from a dirty image.

    Args:
      vis: Measurement set.
      imagename: Image file name.
      nproc: Number of processes.
      tclean_args: Other arguments for tclean.
      thresh_niter: Optional; Number of iterations to find threshhold.
      nsigma: Optional; Threshold level over rms for each iteration.
      sigfig: Optional; Number of significant figures.
      log: Optional; Logging function.
    """
    thresh = None
    for niter in range(thresh_niter):
        # Clean
        if thresh is None:
            # Compute dirty at niter = 0
            clean_args = tclean_args | {'niter': 0}
        tclean_parallel(vis, imagename.with_name(imagename.stem), nproc,
                        clean_args, log=log)

        # Export fits
        fitsimage = f"{imagename}_niter{niter}.fits"
        if clean_args.get('deconvolver', 'hogbom') == 'mtmfs':
            exportfits(f'{imagename}.tt0', fitsimage=f'{fitsimage}',
                       overwrite=True)
        else:
            exportfits(f'{imagename}', fitsimage=f'{fitsimage}',
                       overwrite=True)
        image = fits.open(fitsimage)[0]
        data = image.data * u.Unit(image.header['BUNIT'])

        # Get rms and threshold
        if len(nsigma) == 1:
            nrms = nsigma[0]
        else:
            nrms = nsigma[niter]
        thresh = nrms * mad_std(data, ignore_nan=True)
        thresh = round_sigfig(thresh.to(u.mJy/u.beam), sigfig)

        # Update clean parameters for next iteration
        if log is not None:
            log.info(f'Iteration %i threshold: %s', niter+1, thresh)
        clean_args = {'niter': tclean_args.get('niter', 100000),
                      'threshold': f'{thresh.value}{thresh.unit*u.beam}',
                      'calcpsf': False,
                      'calcres': False}
        clean_args = tclean_args | clean_args

    # Final clean
    tclean_parallel(vis, imagename.with_name(imagename.stem), nproc,
                    clean_args, log=log)

    return f'{thresh.value}{thresh.unit*u.beam}'

def cube_multi_images(imagename: Path) -> Tuple[Path]:
    """Generate image names for `cube_multi_clean`."""
    initial_imagename = imagename.with_suffix('.hogbom.automasking.image')
    final_imagename = imagename.with_suffix('.multiscale.image')

    return initial_imagename, final_imagename

def cube_multi_clean(vis: Path,
                     imagename: Path,
                     nproc: int,
                     array: str,
                     tclean_args: dict,
                     thresh_niter: int = 2,
                     nsigma: Tuple[float] = (3.,),
                     sigfig: int = 3,
                     plot_results: bool = False,
                     resume: bool = False,
                     log: Optional['logging.Logger'] = None) -> Union[Path,
                                                                      str]:
    """Clean cubes in multiple steps.

    It first run a `Hogbom` with `auto-multithresh` clean to produce a mask and
    threshold. Then it uses these and `multiscale` for a final clean.

    Args:
      vis: Measurement set.
      imagename: Image file name.
      nproc: Number of processes.
      array: Array to image.
      tclean_args: Other arguments for tclean.
      thresh_niter: Optional; Number of iterations to find threshhold.
      nsigma: Optional; Threshold level over rms for each iteration.
      sigfig: Optional; Number of significant figures.
      plot_results: Optional; Plot initial clean results.
      resume: Optional; Resume where it was left?
      log: Optional; Logger.

    Returns:
      The final image name and the threshold.
    """
    # Check files
    initial_imagename, final_imagename = cube_multi_images(imagename)
    if ((initial_imagename.exists() and not resume) or
        (initial_imagename.exists() and not final_imagename.exists())):
        if log is not None:
            log.warning('Deleting first iteration cubes: %s',
                        initial_imagename)
        os.system(f"rm -rf {initial_imagename.with_suffix('.*')}")

    # First clean
    clean_args = {'deconvolver': 'hogbom',
                  'usemask': 'auto-multithresh'}
    clean_args = tclean_args | clean_args
    clean_args = recommended_auto_masking(array) | clean_args
    threshold = auto_thresh_clean(vis, initial_imagename, nproc, clean_args,
                                  nsigma=nsigma, log=log)
    mask = initial_imagename.with_suffix('.mask')

    # Plot results of first round
    if plot_results:
        outdir = vis.parent / 'plots'
        outdir.mkdir(exist_ok=True)
        if log is not None:
            log.info('Plotting spectrum')
        plotname = initial_imagename.with_suffix('.cube.spectrum.png')
        plotname = outdir / plotname.name
        chans = plot_imaging_spectra(initial_imagename, plotname,
                                     threshold=threshold,
                                     masking='auto-multithresh')
        if log is not None:
            log.info('Plotting imaging results')
        plotname = initial_imagename.with_suffix('.cube.results.png')
        plotname = outdir / plotname.name
        plot_imaging_products(initial_imagename, plotname, 'hogbom',
                              'auto-multithresh', threshold=threshold,
                              chans=chans)

    # Second clean
    clean_args = {'deconvolver': 'multiscale',
                  'scales': tclean_args.get('scales', [0,5,15]),
                  'threshold': threshold,
                  'usemask': 'user',
                  'mask': f'{mask}'}
    clean_args = tclean_args | clean_args
    tclean_parallel(vis, final_imagename.with_name(final_imagename.stem), nproc,
                    clean_args, log=log)

    return final_imagename, threshold

