"""Utility functions."""
from typing import List, Optional, Sequence, Callable, Dict, Union, Tuple
from inspect import signature
import os

from casatasks import vishead
from casatools import msmetadata, ms
import astropy.units as u
import numpy as np

from .common_types import SectionProxy

def gaussian_primary_beam(freq: u.Quantity, diameter: u.Quantity) -> u.Quantity:
    """Calculate the angular size of a Gaussian primary beam."""
    b = 1 / np.sqrt(np.log(2))
    wvl = freq.to(u.m, equivalencies=u.spectral())
    pb = b * wvl / diameter * u.rad
    return pb.to(u.arcsec)

def gaussian_beam(freq: u.Quantity, baseline: u.Quantity) -> u.Quantity:
    """Estimate the synthetize Gaussian beam size."""
    wvl = freq.to(u.m, equivalencies=u.spectral())
    beam = wvl / baseline * u.rad

    return beam.to(u.arcsec)

def round_sigfig(val: Union[u.Quantity, float],
                 sigfig: int = 1) -> Union[u.Quantity, float]:
    """Round number to given number of significant figures."""
    logval = np.rint(np.log10(val.value))
    newval =  np.round(val.value / 10**logval, sigfig-1) * 10**logval

    return newval * val.unit

def extrema_ms(uvdata: 'pathlib.Path',
               spw: Optional[int] = None) -> Tuple[u.Quantity]:
    """Calculate the frequency range for MS or SPW and longest baseline."""
    mstool = ms()
    mstool.open(f'{uvdata}')
    metadata = mstool.metadata()
    info = mstool.range(['uvdist'])
    max_baseline = np.max(info['uvdist']) * u.m
    if spw is None:
        low_freq = np.inf
        high_freq = 0
        for i in range(metadata.nspw()):
            low_aux = np.min(metadata.chanfreqs(spw=i)) * u.Hz
            high_aux = np.max(metadata.chanfreqs(spw=i)) * u.Hz
            low_freq = min(low_freq, low_aux)
            high_freq = max(high_freq, high_aux)
    else:
        low_freq = np.min(metadata.chanfreqs(spw=spw)) * u.Hz
        high_freq = np.max(metadata.chanfreqs(spw=spw)) * u.Hz

    return low_freq.to(u.GHz), high_freq.to(u.GHz), max_baseline

def get_targets(uvdata: 'pathlib.Path',
                intent: str = 'OBSERVE_TARGET#ON_SOURCE') -> Sequence[str]:
    """Identify targets and return the field names."""
    metadata = msmetadata()
    metadata.open(f'{uvdata}')
    fields = metadata.fieldsforintent(intent, asnames=True)
    metadata.close()

    return fields

def get_array(uvdata: 'pathlib.Path') -> str:
    """Get the ALMA array: 12m, 7m, TP.

    WARNING: TP has not been implemented yet.
    """
    metadata = msmetadata()
    metadata.open(f'{uvdata}')
    diameters = metadata.antennadiameter()
    is12m = [int(val['value']) == 12 for val in diameters.values()]
    is7m = [int(val['value']) == 7 for val in diameters.values()]
    metadata.close()
    if all(is12m):
        return '12m'
    elif all(is7m):
        return '7m'
    else:
        raise ValueError('Cannot identify array')

def find_spws(field: str, uvdata: 'pathlib.Path') -> str:
    """Filter spws for source."""
    metadata = msmetadata()
    metadata.open(f'{uvdata}')
    spws = []
    for spw in metadata.spwsforfield(field):
        if 'FULL_RES' in metadata.namesforspws(spw)[0]:
            spws.append(str(spw))
    metadata.close()

    return ','.join(spws)

def validate_step(in_skip: bool,
                  filename: 'pathlib.Path',
                  log: Callable = print) -> bool:
    """Validate the step.

    If the step is not in skip and `filename` exists, then it is deleted and
    return `True` to run the step again. If the step is in skip but `filename`
    does not exist, then run the step to generate it.
    """
    if in_skip and filename.exists():
        return False
    elif filename.exists():
        log(f'Deleting: {filename}')
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

