"""Utility functions."""
from typing import List, Optional, Sequence, Callable, Dict, Union, Tuple
from inspect import signature
import os

from casatools import msmetadata, ms
import astropy.units as u
import numpy as np

from .common_types import SectionProxy

def get_metadata(uvdata: 'pathlib.Path'):
    """Read metadata from MS."""
    metadata = msmetadata()
    metadata.open(f'{uvdata}')

    return metadata

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

def find_near_exact_denominator(num: int, den: int,
                                direction: str = 'up') -> Tuple[int, int]:
    """Find a closer common denominator to the input one."""
    if num % den == 0:
        return den, num//den
    else:
        if direction == 'up':
            inc = 1
        else:
            inc = -1
        return find_near_exact_denominator(num, den + inc, direction=direction)

def get_spw_start(uvdata: 'pathlib.Path',
                  spws: str,
                  width: Optional[str] = None):
    """Find the common range for the same spw in different EBs."""
    # Check width unit
    if width is not None:
        width_qa = u.Quantity(width)
        spectral_unit = width_qa.unit
    else:
        spectral_unit = u.Hz

    # Load ms information
    mstool = ms()
    mstool.open(f'{uvdata}')
    metadata = mstool.metadata()
    
    # Iterate over spws
    starts = np.array([]) * spectral_unit
    ends = np.array([]) * spectral_unit
    widths = np.array([]) * spectral_unit
    for spw in map(int, spws.split(',')):
        freqs = mstool.cvelfreqs(spwids=[spw], outframe='LSRK') * u.Hz
        if spectral_unit.is_equivalent(u.m/u.s):
            ref_freq = metadata.reffreq(spw)
            ref_freq = ref_freq['m0']['value'] * u.Unit(ref_freq['m0']['unit'])
            freq_to_vel = u.doppler_radio(ref_freq)
            freqs = freqs.to(spectral_unit, equivalencies=freq_to_vel)
        starts.insert(0, np.min(freqs))
        ends.insert(0, np.max(freqs))
        widths.insert(0, np.abs(freqs[0] - freqs[1]))
    start = np.max(starts)
    end = np.min(ends)

    # Set tclean values
    if width is None:
        width_qa = np.mean(widths)
        width = f'{width_qa.value}{width_qa.unit}'
    nchan = np.abs(end - start)/width_qa
    nchan = int(np.floor(nchan.to(1).value))
    start = f'{start.value}{start.unit}'

    return start, nchan, width

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
            freqs = mstool.cvelfreqs(spwids=[i], outframe='LSRK') * u.Hz
            low_aux = np.min(freqs)
            high_aux = np.max(freqs)
            #low_aux = np.min(metadata.chanfreqs(spw=i)) * u.Hz
            #high_aux = np.max(metadata.chanfreqs(spw=i)) * u.Hz
            low_freq = min(low_freq, low_aux)
            high_freq = max(high_freq, high_aux)
    else:
        freqs = mstool.cvelfreqs(spwids=[spw], outframe='LSRK') * u.Hz
        low_freq = np.min(freqs)
        high_freq = np.max(freqs)
        #low_freq = np.min(metadata.chanfreqs(spw=spw)) * u.Hz
        #high_freq = np.max(metadata.chanfreqs(spw=spw)) * u.Hz

    return low_freq.to(u.GHz), high_freq.to(u.GHz), max_baseline

def get_targets(uvdata: 'pathlib.Path',
                intent: str = 'OBSERVE_TARGET#ON_SOURCE') -> Sequence[str]:
    """Identify targets and return the field names."""
    metadata = get_metadata(uvdata)
    fields = metadata.fieldsforintent(intent, asnames=True)
    metadata.close()

    return fields

def get_array(uvdata: 'pathlib.Path') -> str:
    """Get the ALMA array: 12m, 7m, TP.

    WARNING: TP has not been implemented yet.
    """
    metadata = get_metadata(uvdata)
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

def validate_step(resume: bool,
                  filename: 'pathlib.Path',
                  log: Callable = print) -> bool:
    """Validate the step.

    If the step is not in skip and `filename` exists, then it is deleted and
    return `True` to run the step again. If the step is in skip but `filename`
    does not exist, then run the step to generate it.
    """
    if resume and filename.exists():
        return False
    elif filename.exists():
        log(f'Deleting: {filename}')
        os.system(f'rm -rf {filename}')

    return True

def max_chan_width(freq: u.Quantity,
                   diameter: u.Quantity,
                   max_baseline: u.Quantity,
                   reduction: float = 0.99) -> u.Quantity:
    """Calculate the maximum channel width.

    Given the desired reduction in peak response value, it calculates the
    maximum channel width accepted. The maximum channel width is calculated by
    solving the equations in [CASA
    guides](https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf).

    Args:
      freq: frequency.
      diameter: antenna diameter.
      max_baseline: maximum baseline.
      reduction: optional; reduction in peak response.

    Returns:
      Maximum channel width using a Gaussian response.
    """
    chan_width = (freq * 2 * np.sqrt(np.log(2)) * diameter / max_baseline *
                  np.sqrt(1/reduction**2 - 1))

    return chan_width.to(u.MHz)

def continuum_bins(uvdata: 'pathlib.Path',
                   diameter: u.Quantity,
                   log: Callable = print) -> Sequence[int]:
    """Find the maximum continuum bin sizes."""
    metadata = get_metadata(uvdata)
    bandwidths = metadata.bandwidths()
    bins = []
    for i in range(metadata.nspw()):
        low_freq, _, baseline = extrema_ms(uvdata, spw=i)
        max_width = max_chan_width(low_freq, diameter, baseline)
        ngroups = bandwidths[i] * u.Hz / max_width
        ngroups = int(np.ceil(ngroups.to(1).value))
        if ngroups <= 1:
            binsize = metadata.nchan(i)
        else:
            ngroups, binsize = find_near_exact_denominator(metadata.nchan(i),
                                                           ngroups)
        bins.append(binsize)

        # Log results
        log(f'SPW: {i}')
        log(f'Lowest frequency: {low_freq}')
        log(f'Longest baseline: {baseline}')
        log(f'Maximum bin width: {max_width.to(u.MHz)}')
        log((f'Bandwidth: {bandwidths[i]/1E6} MHz '
             f'({ngroups} of {binsize} channels)'))
    metadata.close()

    return bins

def flags_from_cont_ranges(ranges: Dict[int, List[str]],
                           uvdata: 'pathlib.Path',
                           invert: bool = False,
                           mask_borders: bool = False,
                           border: int = 10) -> Dict[int, List[Tuple[int]]]:
    """Convert continuum ranges into channel flags.
    
    It uses input continuum ranges in LSRK frequency and convert them into
    channel flags. The flags correspond to channels that will be flagged to
    compute the continuum.

    If `invert` is set, then the output channel ranges will correspond to
    channels that are not flagged.

    Args:
      ranges: Ranges of continuum frequencies per SPW.
      uvdata: UV data to extract the frequency axis.
      invert: Optional; Return unflagged data?
      mask_borders: Optional; Mask (flag) channels at the borders?
      border: Optional; Amount of channels to mask (flag).

    Returns:
      A dictionary associating each SPW with flagged channel ranges.
    """
    mstool = ms()
    mstool.open(f'{uvdata}')
    flags_chan = {}
    for spw, rng_spw in ranges.items():
        # Convert to masked array
        freqs = mstool.cvelfreqs(spwids=[spw], outframe='LSRK') * u.Hz
        freqs = np.ma.array(freqs.to(u.GHz).value)

        # Mask the frequencies
        for rng in rng_spw:
            freq_ran = list(map(u.Quantity, rng.split()[0].split('~')))
            freq_ran[0] = freq_ran[0] * freq_ran[1].unit
            freq_ran = freq_ran[0].to(u.GHz).value, freq_ran[1].to(u.GHz).value
            freqs = np.ma.masked_inside(freqs, *freq_ran)

        # Invert channel selection to convert into flags
        freqs = np.ma.array(freqs.data, mask=~freqs.mask)

        # Mask borders
        if mask_borders:
            freqs[:border] = np.ma.masked
            freqs[-border:] = np.ma.masked

        # Clump (un)masked data
        if invert:
            clumps = np.ma.clump_unmasked(freqs)
        else:
            clumps = np.ma.clump_masked(freqs)

        # Convert to indices
        clump_ind = []
        for clump in clumps:
            clump_ind.append((int(clump.start), int(clump.stop-1)))
        flags_chan[spw] = clump_ind

    # Close tool
    mstool.close()

    return flags_chan

def clumps_to_casa(clumps_per_spw: Dict[int, List[Tuple[int]]]) -> str:
    """Convert ranges per SPW into casa format."""
    ranges = []
    for spw, clumps in clumps_per_spw.items():
        spw_ranges = map(lambda x: f'{x[0]}~{x[1]}', clumps)
        ranges.append(f'{spw}:' + ';'.join(spw_ranges))

    return ','.join(ranges)

def find_spws(uvdata: 'pathlib.Path',
              intent: str = 'OBSERVE_TARGET#ON_SOURCE') -> str:
    """Filter science spws for source."""
    metadata = get_metadata(uvdata)
    spws = []
    for spw in metadata.spwsforintent(intent):
        if 'FULL_RES' in metadata.namesforspws(spw)[0]:
            spws.append(str(spw))
    metadata.close()

    return ','.join(spws)

#def get_spws(vis: 'pathlib.Path') -> List:
#    """Retrieve the spws in a visibility ms."""
#    return vishead(vis=str(vis), mode='get', hdkey='spw_name')[0]

def get_spws_indices(uvdata: 'pathlib.Path',
                     selected: Optional[Sequence[str]] = None) -> List[str]:
    """Get the indices of the spws.

    This task search for duplicated spws (e.g. after measurement set
    concatenation) and groups the spws indices.

    Args:
      vis: Measurement set.
      selected: optional; Selected spectral windows.

    Returns:
      A list with the spws in the vis ms.
    """
    # Open meta data
    metadata = get_metadata(uvdata)

    # Create dictionary indexing spws with (baseband, subwin)
    names_spw = metadata.spwsfornames()
    bb_info = {}
    for name, ind in names_spw.items():
        aux = name.split('#')
        key = aux[2].split('_')[1], aux[3].split('-')[1]
        val = bb_info.get(key, [])
        val += list(ind)
        bb_info[key] = sorted(val)

    # Extract spws
    spws = sorted(bb_info.values(), key=lambda x: x[0])
    metadata.close()

    if selected is not None:
        return [','.join(map(str, spw))
                for i, spw in enumerate(spws)
                if f'{i}' in selected]
    else:
        return [','.join(map(str, spw)) for spw in spws]

    # Spectral windows names
    # Extract name by BB_XX and subwin SW_YY
    #names = [aux.split('#')[2]+'_'+aux.split('#')[3] for aux in get_spws(vis)]
    #name_set = set(names)

    ## Cases
    #if len(name_set) != len(names):
    #    spwinfo = {}
    #    for i, key in enumerate(names):
    #        if key not in spwinfo:
    #            spwinfo[key] = f'{i}'
    #        else:
    #            spwinfo[key] += f',{i}'
    #else:
    #    spwinfo = dict(zip(names, map(str, range(len(names)))))

    ## Spectral windows indices
    #if spws is not None:
    #    spw_ind = list(map(int, spws))
    #else:
    #    spw_ind = list(range(len(name_set)))

    #return [spw for i, spw in enumerate(spwinfo.values()) if i in spw_ind]

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

