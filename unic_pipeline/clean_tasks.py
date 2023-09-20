"""Tools for running the `tclean` CASA task."""
#from typing import Any, Callable, List, Optional, Sequence, TypeVar, Mapping
from typing import Callable, Sequence
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess

from casatasks import tclean

from .common_types import SectionProxy
from .utils import get_func_params

def get_tclean_params(
    config: SectionProxy,
    required_keys: Sequence[str] = ('cell', 'imsize'),
    ignore_keys: Sequence[str] = ('vis', 'imagename', 'spw'),
    float_keys: Sequence[str]  = ('robust', 'pblimit', 'pbmask'),
    int_keys: Sequence[str] = ('niter', 'chanchunks'),
    bool_keys: Sequence[str] = ('interactive', 'parallel', 'pbcor',
                                'perchanweightdensity'),
    int_list_keys: Sequence[str] = ('imsize', 'scales'),
    cfgvars: Optional[Dict] = None,
) -> dict:
    """Filter input parameters and convert values to the correct type.

    Args:
      config: `ConfigParser` section proxy with input parameters to filter.
      ignore_keys: optional; tclean parameters to ignore.
      float_keys: optional; tclean parameters to convert to float.
      int_keys: optional; tclean parameters to convert to int.
      bool_keys: optional; tclean parameters to convert to bool.
      int_list_keys: optional; tclean parameters as list of integers.
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
                    log: Callable = print):
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
        tclean(vis=str(vis), imagename=str(imagename), **tclean_args)
    else:
        # Save tclean params
        tclean_args.update({'parallel': True})
        paramsfile = imagename.parent / 'tclean_params.json'
        paramsfile.write_text(json.dumps(tclean_args, indent=4))

        # Run
        cmd = os.environ.get('MPICASA', f'mpicasa -n {nproc} casa')
        script = Path(__file__).parent / 'run_tclean_parallel.py'
        logfile = datetime.now().isoformat(timespec='milliseconds')
        logfile = f'tclean_parallel_{logfile}.log'
        cmd = (f'{cmd} --nogui --logfile {logfile} '
               f'-c {script} {vis} {imagename} {paramsfile}')
        log(f'Running: {cmd}')
        # pylint: disable=R1732
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        proc.wait()

