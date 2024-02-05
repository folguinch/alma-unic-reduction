"""Tools for working with spectrum data."""
from typing import Tuple, Callable, Optional, List, Dict

from casatools import ms
from scipy import stats
from spectral_cube import SpectralCube
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def extract_spectrum(image: 'pathlib.Path') -> Tuple[u.Quantity]:
    """Extract spectrum at the position of the maximum."""
    # Open cube
    cube = SpectralCube.read(image)
    cube.allow_huge_operations = True
    cube = cube.with_spectral_unit(u.GHz, velocity_convention='radio')

    # Collapse image
    collapsed = np.sum(cube.unmasked_data[:], axis=0)
    loc = np.unravel_index(np.nanargmax(collapsed), collapsed.shape)

    # Spectrum
    freq = cube.spectral_axis
    flux = cube.unmasked_data[:, loc[0], loc[1]]

    return freq, flux

def freq_to_chan(freq: npt.ArrayLike, chan: npt.ArrayLike) -> Tuple[Callable]:
    """Generate functions to convert frequencies in channels and viceversa."""
    slope, interp, *_ = stats.linregress(freq, chan)

    return lambda x: slope * x + interp, lambda x: (x - interp) / slope

def plot_spectrum(spectrum: Tuple[u.Quantity],
                  filename: 'pathlib.Path',
                  selection: Optional[List[Tuple[int]]] = None,
                  functions: Tuple[Callable] = None):
    """Plot a spectrum."""
    # Figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot spectrum
    ax.plot(spectrum[0].value, spectrum[1].to(u.mJy/u.beam).value, 'b-')
    ax.set_xlim(np.min(spectrum[0].value), np.max(spectrum[0].value))
    ax.set_xlabel('LSRK Frequency (GHz)')
    ax.set_ylabel('Intensity (mJy/beam)')

    # Axes properties
    if functions is not None:
        ax_top = ax.secondary_xaxis('top', functions=functions)
        ax_top.set_xlabel('Channels')

    # Selection
    if selection is not None:
        opts = {'color': 'r', 'alpha': 0.2}
        for selected in selection:
            ax.axvspan(*selected, **opts)

    # Save
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_spectral_selection(spectra: List[Tuple[u.Quantity]],
                            selection: Dict[int, List[Tuple[int]]],
                            uvdata: 'pathlib.Path',
                            spws: List[str],
                            outdir: 'pathlib.Path',
                            suffix: str = ''):
    """Plot spectrum and data selection."""
    mstool = ms()
    mstool.open(f'{uvdata}')
    for spw, spec in zip(spws, spectra, strict=True):
        for subspw in map(int, spw.split(',')):
            # Spectral axis of uvdata
            spectral_axis = mstool.cvelfreqs(spwids=[subspw],
                                             outframe='LSRK') * u.Hz
            spectral_axis = spectral_axis.to(u.GHz)
            functions = freq_to_chan(spectral_axis.value,
                                     np.indices(spectral_axis.shape))

            # Convert selection to frequencies
            freq_selection = [(spectral_axis[ind0].value,
                               spectral_axis[ind1].value)
                              for ind0, ind1 in selection[subspw]]

            # Plot data
            new_suffix = suffix + f'.spw{subspw}.png'
            filename = uvdata.with_suffix(new_suffix).name
            filename = outdir / filename
            plot_spectrum(spec, filename, selection=freq_selection,
                          functions=functions)
