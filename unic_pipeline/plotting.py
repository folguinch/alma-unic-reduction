"""Plotting tools."""
from typing import Optional, Tuple, Dict
from itertools import product

from astropy.io import fits
from astropy.stats import mad_std
from casatasks import exportfits
from spectral_cube import SpectralCube
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from .spectrum_tools import extract_spectrum, plot_spectrum

def plot_imaging_products(imagename: 'pathlib.Path',
                          filename: 'pathlib.Path',
                          deconvolver: str,
                          mask_type: str,
                          threshold: Optional[str] = None,
                          chans: Optional[Dict[str, int]] = None):
    """Plot imaging results."""
    # Set image types
    imtypes = ('.image', '.residual')
    if mask_type == 'auto-multithresh':
        imtypes += ('.autothresh1', '.mask')

    # Determine additional suffix
    suffix = ''
    if deconvolver == 'mtmfs':
        suffix = '.tt0'

    # Export fits and store images and masks
    images = []
    masks = []
    for imtype in imtypes:
        if imtype in ['.image', '.residual']:
            imname = imagename.with_suffix(f'{imtype}{suffix}')
            fitsimage = imagename.with_suffix(f'{imtype}{suffix}.fits')
            images.append(fitsimage)
        else:
            imname = imagename.with_suffix(f'{imtype}')
            fitsimage = imagename.with_suffix(f'{imtype}.fits')
            masks.append(fitsimage)
        if not fitsimage.exists():
            exportfits(imagename=f'{imname}', fitsimage=f'{fitsimage}')
    
    # Plot data
    if chans is None:
        chans = {'dummy': None}
    for description, chan in chans.items():
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True,
                                figsize=(15, 8))
        for i, (ax, imname) in enumerate(zip(axs, images)):
            # Plot image
            image = np.squeeze(fits.open(imname)[0].data)
            if chan is not None:
                image = image[chan,:,:]
            vmin = np.nanmin(image)
            vmax = np.nanmax(image)
            ax.imshow(image, vmin=vmin, vmax=vmax, origin='lower',
                      cmap='viridis')

            # Contours
            rms = mad_std(image, ignore_nan=True)
            if i == 0:
                contour_map = np.squeeze(image)
                levels = []
                n = 3
                while n * rms < vmax:
                    levels.append(n * rms)
                    n = n * 2
            ax.contour(contour_map, levels=levels, colors='k',  origin='lower',
                       linewidths=1)
            if threshold is not None:
                level = u.Quantity(threshold).to(u.Jy).value
                ax.contour(contour_map, levels=[level], colors='m',
                           origin='lower', linewidths=1)
            
            # Decorations
            if i == 0:
                title = (f'Peak: {vmax*1E3:.1f}mJy/beam '
                         f'- rms: {rms*1E6:.1f}uJy/beam '
                         f'- SN: {int(vmax/rms)}')
            else:
                title = (f'Peak: {vmax*1E6:.1f}uJy/beam '
                         f'- rms: {rms*1E6:.1f}uJy/beam')
            ax.set_title(title)

            # Limits
            invalid = np.isnan(np.squeeze(image))
            invalid = ~np.all(invalid, axis=0)
            ind = np.arange(len(invalid))
            lim1, lim2 = min(ind[invalid]), max(ind[invalid])
            ax.set_xlim(lim1, lim2)
            ax.set_ylim(lim1, lim2)

            for mask, color in zip(masks, ['#da4979','#8dd1dc']):
                maskimage = np.squeeze(fits.open(mask)[0].data)
                if chan is not None:
                    maskimage = maskimage[chan,:,:]
                ax.contour(maskimage.astype(int), levels=[0.999],
                           colors=color, origin='lower', linewidths=1)

        # Save image
        if chan is not None:
            plotname = filename.with_suffix((f'.chan{chan}'
                                             f'.{description}'
                                             f'{filename.suffix}'))
        else:
            plotname = filename
        fig.savefig(plotname, bbox_inches='tight')
        plt.close()

def plot_comparison(reference: 'pathlib.Path',
                    imagename: 'pathlib.Path',
                    filename: 'pathlib.Path'):
    """Plot a comparison between images."""
    # Determine real file names
    images = []
    for img in [reference, imagename]:
        if not img.exists():
            if img.with_suffix('.image.tt0').exists():
                imname = img.with_suffix('.image.tt0')
            elif img.suffix == '.fits':
                imname = img.parent / img.stem
                if not imname.exists():
                    raise IOError(f'Cannot find file: {img}')
            else:
                raise IOError(f'Cannot find file: {img}')
        elif img.suffix == '.fits':
            images.append(fits.open(img)[0].data)
            continue
        else:
            imname = img
        fitsimage = imname.with_suffix(f'{imname.suffix}.fits')
        if not fitsimage.exists():
            exportfits(imagename=f'{imname}', fitsimage=f'{fitsimage}')
        images.append(fits.open(fitsimage)[0].data)

    # Data limits
    vmax = max([np.nanmax(img) for img in images])
    rms = [mad_std(img, ignore_nan=True) for img in images]
    vmin = -2 * min(rms)

    # Difference
    images.append(np.absolute(images[0] - images[1]))

    # Plot data
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True,
                            figsize=(25, 10))
    for i, (ax, image) in enumerate(zip(axs, images)):
        # Plot image
        if i != 2:
            ax.imshow(np.squeeze(image), vmin=vmin, vmax=vmax,
                      origin='lower', cmap='viridis')
        else:
            ax.imshow(np.squeeze(image), vmin=np.nanmin(image),
                      vmax=np.nanmax(image), origin='lower', cmap='viridis')

        # Contours
        if i != 2:
            contour_map = np.squeeze(image)
            minrms = min(rms)
            levels = []
            n = 3
            while n * minrms < vmax:
                levels.append(n * minrms)
                n = n * 2
            ax.contour(contour_map, levels=levels, colors='k',  origin='lower',
                       linewidths=1)
        else:
            ax.contour(contour_map, levels=levels, colors='k',  origin='lower',
                       linewidths=1)
        
        # Decorations
        if i != 2:
            title = (f'Peak: {np.nanmax(image)*1E3:.1f}mJy/beam - '
                     f'rms: {rms[i]*1E6:.1f}uJy/beam'
                     f' - SN: {int(np.nanmax(image)/rms[i])}')
        else:
            title = f'Peak: {np.nanmax(image)*1E6:.1f}uJy/beam'
        ax.set_title(title)

        # Limits
        invalid = np.isnan(np.squeeze(image))
        invalid = ~np.all(invalid, axis=0)
        ind = np.arange(len(invalid))
        lim1, lim2 = min(ind[invalid]), max(ind[invalid])
        ax.set_xlim(lim1, lim2)
        ax.set_ylim(lim1, lim2)

    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_imaging_spectra(imagename: 'pathlib.Path',
                         plotname: 'pathlib.Path',
                         threshold: Optional[str] = None,
                         masking: Optional[str] = None) -> Dict[str, int]:
    """Plot spectra and mask."""
    # Images to plot
    freq, flux, loc = extract_spectrum(imagename)
    residual = extract_spectrum(imagename.with_suffix('.residual'), loc=loc)[1]
    residual = residual * u.Jy / u.beam
    masks = []
    filenames = []
    if masking == 'user':
        masks.append(extract_spectrum(imagename.with_suffix('.mask'),
                                      loc=loc)[1])
        filenames.append(plotname.with_suffix((f'.x{loc[1]}y{loc[0]}'
                                               f'{plotname.suffix}')))
    elif masking == 'auto-multithresh':
        i = 1
        while True:
            mask = imagename.with_suffix(f'.autothresh{i}')
            if mask.exists():
                mask = extract_spectrum(mask, loc=loc)[1]
                masks.append(mask)
            else:
                break
            filenames.append(plotname.with_suffix((f'.x{loc[1]}y{loc[0]}'
                                                   f'.autothresh{i}'
                                                   f'{plotname.suffix}')))
            i += 1
    else:
        filenames.append(plotname.with_suffix((f'.x{loc[1]}y{loc[0]}'
                                               f'{plotname.suffix}')))
    
    # Plot spectrum
    chans = {}
    title_base = f'Position: (x,y) = ({loc[1]},{loc[0]})'
    for i, filename in enumerate(filenames):
        # Get selection and channel with max value outside mask
        title = title_base + ' '
        if len(masks) != 0:
            # Selection
            mask_freq = np.ma.array(freq, mask=masks[i])
            indx = np.ma.clump_masked(mask_freq)
            selection = [(freq[sel][0].value, freq[sel][-1].value)
                         for sel in indx]
            title += f'- Valid channels: {np.sum(masks[i])}/{len(masks[i])}'

            # Find channel
            masked_flux = np.ma.array(flux.value, mask=masks[i])
            chans['max_out_mask'] = np.nanargmax(masked_flux)
        else:
            selection = None

        # Plot
        if threshold is not None:
            hline = u.Quantity(threshold) / u.beam
            title += f' - Threshold: {threshold}'
        else:
            hline = None
        plot_spectrum((freq, flux), filename, residual=(freq, residual),
                      selection=selection, title=title, hline=hline)

    # Select channels for image plane plots
    peak = np.nanmax(flux.value)
    masked_flux1 = np.ma.masked_where(flux.value < peak/2, flux.value)
    masked_flux2 = np.ma.masked_where(flux.value < peak/3, flux.value)
    chans['peak'] = np.nanargmax(flux.value)
    chans['half_max'] = np.nanargmin(masked_flux1)
    chans['third_max'] = np.nanargmin(masked_flux2)

    return chans
