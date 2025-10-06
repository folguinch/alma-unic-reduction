"""Tools for working with spectrum data."""
from typing import Tuple, Callable, Optional, List, Dict
import warnings

from casatools import ms
from scipy import stats
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def extract_spectrum(image: 'pathlib.Path',
                     loc: Optional[Tuple[int]] = None) -> Tuple[u.Quantity]:
    """Extract spectrum at the position of the maximum."""
    # Disable warnings
    warnings.filterwarnings(action='ignore', category=SpectralCubeWarning,
                            append=True)

    # Open cube
    if image.suffix == '.fits':
        dformat = 'fits'
        use_dask = False
    else:
        dformat = 'casa'
        use_dask = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cube = SpectralCube.read(image, format=dformat, use_dask=use_dask)
        cube.allow_huge_operations = True
        cube = cube.with_spectral_unit(u.GHz, velocity_convention='radio')

    # Collapse image
    if loc is None:
        collapsed = np.nansum(cube.unmasked_data[:], axis=0)
        loc = np.unravel_index(np.nanargmax(collapsed), collapsed.shape)

    # Spectrum
    freq = cube.spectral_axis
    flux = cube.unmasked_data[:, loc[0], loc[1]]

    return freq, flux, loc

def freq_to_chan(freq: npt.ArrayLike, chan: npt.ArrayLike) -> Tuple[Callable]:
    """Generate functions to convert frequencies in channels and viceversa."""
    slope, interp, *_ = stats.linregress(freq, chan)

    return lambda x: slope * x + interp, lambda x: (x - interp) / slope

def plot_spectrum(spectrum: Tuple[u.Quantity],
                  filename: 'pathlib.Path',
                  residual: Optional[Tuple[u.Quantity]] = None,
                  selection: Optional[List[Tuple[int]]] = None,
                  functions: Optional[Tuple[Callable]] = None,
                  title: str = '',
                  hline: Optional[u.Quantity] = None):
    """Plot a spectrum."""
    # Figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot spectrum
    ax.plot(spectrum[0].value, spectrum[1].to(u.mJy/u.beam).value, 'b-')
    if residual is not None:
        ax.plot(residual[0].value, residual[1].to(u.mJy/u.beam).value, 'g-')
    ax.set_xlim(np.min(spectrum[0].value), np.max(spectrum[0].value))
    ax.set_xlabel('LSRK Frequency (GHz)')
    ax.set_ylabel('Intensity (mJy/beam)')
    ax.set_title(title)

    # Axes properties
    if functions is not None:
        ax_top = ax.secondary_xaxis('top', functions=functions)
        ax_top.set_xlabel('Channels')

    # Selection
    if selection is not None:
        opts = {'color': 'r', 'alpha': 0.2}
        for selected in selection:
            ax.axvspan(*selected, **opts)

    # Horizontal line
    if hline is not None:
        ax.axhline(y=hline.to(u.mJy/u.beam).value, color='m')

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
