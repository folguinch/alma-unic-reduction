"""Tools for running the `tclean` CASA task."""
from typing import Sequence, Optional, Dict
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess

from casatasks import tclean

from .common_types import SectionProxy
from .utils import get_func_params

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
      ignore_keys: optional; tclean parameters to ignore.
      float_keys: optional; tclean parameters to convert to float.
      int_keys: optional; tclean parameters to convert to int.
      bool_keys: optional; tclean parameters to convert to bool.
      int_list_keys: optional; tclean parameters as list of integers.
      cfgvars: optional; parameters to replace those in config file.
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
      vis: measurement set.
      imagename: image file name.
      nproc: number of processes.
      tclean_args: other arguments for tclean.
      log: optional; logging function.
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

