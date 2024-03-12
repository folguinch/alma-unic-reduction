"""Plotting tools."""
from astropy.io import fits
from astropy.stats import mad_std
from casatasks import exportfits
import matplotlib.pyplot as plt
import numpy as np

def plot_imaging_products(imagename: 'pathlib.Path',
                          filename: 'pathlib.Path',
                          deconvolver: str,
                          mask_type: str):
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
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True,
                            figsize=(15, 8))
    for i, (ax, imname) in enumerate(zip(axs, images)):
        # Plot image
        image = fits.open(imname)[0]
        vmin = np.nanmin(image.data)
        vmax = np.nanmax(image.data)
        ax.imshow(np.squeeze(image.data), vmin=vmin, vmax=vmax, origin='lower',
                  cmap='viridis')

        # Contours
        rms = mad_std(image.data, ignore_nan=True)
        if i == 0:
            contour_map = np.squeeze(image.data)
            levels = []
            n = 3
            while n * rms < vmax:
                levels.append(n * rms)
                n = n * 2
        ax.contour(contour_map, levels=levels, colors='k',  origin='lower',
                   linewidths=1)
        
        # Decorations
        if i == 0:
            title = f'Peak: {vmax*1E3:.1f}mJy/beam - rms: {rms*1E6:.1f}uJy/beam'
            title += f' - SN: {int(vmax/rms)}'
        else:
            title = f'Peak: {vmax*1E6:.1f}uJy/beam - rms: {rms*1E6:.1f}uJy/beam'
        ax.set_title(title)

        # Limits
        invalid = np.isnan(np.squeeze(image.data))
        invalid = ~np.all(invalid, axis=0)
        ind = np.arange(len(invalid))
        lim1, lim2 = min(ind[invalid]), max(ind[invalid])
        ax.set_xlim(lim1, lim2)
        ax.set_ylim(lim1, lim2)

        for mask, color in zip(masks, ['#da4979','#8dd1dc']):
            maskimage = fits.open(mask)[0]
            ax.contour(np.squeeze(maskimage.data.astype(int)), levels=[0.999],
                       colors=color, origin='lower', linewidths=1)

    fig.savefig(filename, bbox_inches='tight')
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
