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
