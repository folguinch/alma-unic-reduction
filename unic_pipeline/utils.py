"""Utility functions."""
from typing import List, Optional, Sequence, Callable, Dict
from inspect import signature
import os

from casatasks import vishead

from .common_types import SectionProxy

def validate_step(in_skip: bool, filename: 'pathlib.Path') -> bool:
    """Validate the step.

    If the step is not in skip and `filename` exists, then it is deleted and
    return `True` to run the step again. If the step is in skip but `filename`
    does not exist, then run the step to generate it.
    """
    if in_skip and filename.exists():
        return False
    elif filename.exists():
        os.system(f'rm -rf {filename}')

    return True

def get_spws(vis: 'pathlib.Path') -> List:
    """Retrieve the spws in a visibility ms."""
    return vishead(vis=str(vis), mode='get', hdkey='spw_name')[0]

def get_spws_indices(vis: 'pathlib.Path',
                     spws: Optional[Sequence[str]] = None) -> List:
    """Get the indices of the spws.

    This task search for duplicated spws (e.g. after measurement set
    concatenation) and groups the spws indices.

    Args:
      vis: measurement set.
      spws: optional; selected spectral windows.

    Returns:
      A list with the spws in the vis ms.
    """
    # Spectral windows names
    # Extract name by BB_XX and subwin SW_YY
    names = [aux.split('#')[2]+'_'+aux.split('#')[3] for aux in get_spws(vis)]
    name_set = set(names)

    # Cases
    if len(name_set) != len(names):
        spwinfo = {}
        for i, key in enumerate(names):
            if key not in spwinfo:
                spwinfo[key] = f'{i}'
            else:
                spwinfo[key] += f',{i}'
    else:
        spwinfo = dict(zip(names, map(str, range(len(names)))))

    # Spectral windows indices
    if spws is not None:
        spw_ind = list(map(int, spws))
    else:
        spw_ind = list(range(len(name_set)))

    return [spw for i, spw in enumerate(spwinfo.values()) if i in spw_ind]

def get_func_params(func: Callable,
                    config: SectionProxy,
                    required_keys: Sequence[str] = (),
                    ignore_keys: Sequence[str] = (),
                    float_keys: Sequence[str] = (),
                    int_keys: Sequence[str] = (),
                    bool_keys: Sequence[str] = (),
                    int_list_keys: Sequence[str] = (),
                    float_list_keys: Sequence[str] = (),
                    sep: str = ',',
                    cfgvars: Optional[Dict] = None,
                    ) -> Dict:
    """Check the input parameters of a function and convert.
    
    Args:
      func: function to inspect.
      config: configuration parser section proxy.
      required_keys: optional; required keys.
      ignore_keys: optional; keys to ignore.
      float_keys: optional; keys required as float type.
      int_keys: optional; keys required as int type.
      bool_keys: optional; keys required as bool type.
      int_list_keys: optional; keys required as list of integers.
      float_list_keys: optional; keys required as list of floats.
      sep: optional; separator for the values in list keys.
      cfgvars: optional; values replacing those in config.
    """
    # Default values
    if cfgvars is None:
        cfgvars = {}

    # Check required arguments are in config
    for opt in required_keys:
        if opt not in config and opt not in cfgvars:
            raise KeyError(f'Missing {opt} in configuration')

    # Filter paramters
    pars = {}
    for key in signature(func).parameters:
        if (key not in config and key not in cfgvars) or key in ignore_keys:
            continue
        # Check for type:
        if key in float_keys:
            pars[key] = config.getfloat(key, vars=cfgvars)
        elif key in int_keys:
            pars[key] = config.getint(key, vars=cfgvars)
        elif key in bool_keys:
            pars[key] = config.getboolean(key, vars=cfgvars)
        else:
            val = config.get(key, vars=cfgvars)
            if key in int_list_keys:
                pars[key] = list(map(int, val.split(sep)))
            elif key in float_list_keys:
                pars[key] = list(map(float, val.split(sep)))
            else:
                pars[key] = val

    return pars

