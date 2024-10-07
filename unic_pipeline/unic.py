#!/bin/python3
"""ALMA-UNIC pipeline.

The pipeline is designed to process data for the ALMA-UNIC LP. It performs
several steps:

  - `split`: It prepares the data starting from the original calibrated data or
    from configuration files. It takes the original data, finds sources for
    science and splits these data into individual MS per array (7m, 12m). It
    concatenates data with multiple EBs. If started from `uvdata`, then it
    generates a configuration file per array collecting information per step
    using the default configuration for ALMA-UNIC as a skeleton.
  - `dirty_cubes`: Calculate dirty cubes from the split data.
  - `continuum`: It takes a `cont.dat` (or similar) file and extracts the
    frequency ranges for continuum calculation. It then converts it to channel
    ranges of flagged channels (channel with lines). Additionally, it
    determines an optimum channel binning for averaging, which are then used to
    compute the continuum visibilities. Two sets of visibilities are produced,
    a line-free set (files ending in `.cont.ms`) and set without line flagging
    (files ending in `.cont_all.ms`).
  - `contsub`: It uses the frequency ranges for continuum calculation to
    compute continuum subtracted visibilities.
  - `combine_arrays`: If 7m and 12m data are given, then a concatenated array
    is produced for the line-free continuum and continuum subtracted
    visibilities. It also generates a configuration file for the combined
    array.
  - `clean_cont`: Clean continuum to produce science-ready images.
"""
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import argparse
import sys

from unic_pipeline.data_handler import DataManager
import unic_pipeline.argparse_parents as parents
import unic_pipeline.argparse_actions as actions

def prep_data(args: argparse.Namespace) -> None:
    """Prepare data for processing."""
    if args.configfiles is not None:
        args.data = DataManager.from_configs(
            args.configfiles,
            args.log,
            datadir=args.basedir[0],
            cont=args.cont,
            resume='split' in args.skip or args.resume,
        )
    elif args.uvdata is not None:
        args.data = DataManager.from_uvdata(
            args.uvdata,
            args.defconfig[0],
            args.log,
            datadir=args.basedir[0],
            cont=args.cont,
            resume='split' in args.skip or args.resume,
        )
    else:
        raise ValueError('Need input data')

def dirty_cubes(args: argparse.Namespace):
    """Calculate dirty cubes."""
    for data in args.data.values():
        data.array_imaging('dirty_cubes', '', nproc=args.nproc[0],
                           per_spw=True, get_spectra=True, niter=0)

def continuum(args: argparse.Namespace):
    """Calculate and image the continuum visibilities."""
    for data in args.data.values():
        data.continuum_visibilities(control_image=True, nproc=args.nproc[0])

def contsub(args: argparse.Namespace):
    """Calculate the continuum subtracted visibilities."""
    for data in args.data.values():
        data.contsub_visibilities(dirty_images=True, nproc=args.nproc[0])

def combine_arrays(args: argparse.Namespace):
    """Combine 7m and 12m data."""
    for data in args.data.values():
        if '7m' not in data.arrays or '12m' not in data.arrays:
            args.log.warning('Arrays missing for combiantion: %s', data.arrays)
            continue
        data.combine_arrays(('7m', '12m'), default_config=args.defconfig[0],
                            datadir=args.basedir[0], control_images=True,
                            nproc=args.nproc[0])

def clean_continuum(args: argparse.Namespace):
    """Clean continuum for different robust and arrays."""
    robust_values = [-2.0, 0.5, 2.0]
    #robust_values = [0.5]
    for data in args.data.values():
        for robust in robust_values:
            # Clean data
            if robust == 0.5:
                compare_to = 'continuum_control'
            else:
                compare_to = None
            data.array_imaging('continuum',
                               'continuum',
                               nproc=args.nproc[0],
                               auto_threshold=True,
                               export_fits=True,
                               plot_results=True,
                               compare_to=compare_to,
                               robust=robust)

def clean_cubes(args: argparse.Namespace):
    """Clean continuum for different robust and arrays."""
    for data in args.data.values():
        data.array_imaging('clean_cubes',
                           'uvcontsub',
                           nproc=args.nproc[0],
                           auto_threshold=True,
                           export_fits=True,
                           per_spw=True,
                           plot_results=True,
                           )

def unic(args: Optional[List] = None) -> None:
    """Run the main UNIC pipeline."""
    # Pipeline steps and functions
    steps = {
        'split': prep_data,
        'dirty_cubes': dirty_cubes,
        'continuum': continuum,
        'contsub': contsub,
        'combine_arrays': combine_arrays,
        'clean_cont': clean_continuum,
        'clean_cubes': clean_cubes,
    }

    # Default config
    default_config = (Path(__file__).resolve().parent /
                      'configs/default.cfg')

    # Command line options
    logfile = datetime.now().isoformat(timespec='milliseconds')
    logfile = f'debug_unic_{logfile}.log'
    args_parents = [parents.logger(logfile)]
    parser = argparse.ArgumentParser(
        description='Execute UNIC pipeline from command line inputs',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('-b', '--basedir', default=[Path('./').resolve()],
                        nargs=1, action=actions.NormalizePath,
                        help='Base directory')
    parser.add_argument('-n', '--nproc', nargs=1, type=int, default=[5],
                        help='Number of processes for parallel processing')
    parser.add_argument('-neb', nargs=1, type=int, default=None,
                        help='Number of EBs')
    parser.add_argument('--skip', nargs='*', choices=list(steps.keys()),
                        default=[],
                        help='Steps to skip')
    # To be implemented
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument('--continuum', action='store_true',
    #                   help='Image only the continuum')
    #group.add_argument('--cubes', action='store_true',
    #                   help='Image only the cubes')
    parser.add_argument('--field', nargs=1, default=None,
                        help='Field name')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already created data.')
    parser.add_argument('--uvdata', nargs='+', type=str,
                        action=actions.NormalizePath,
                        help='uv data ms')
    parser.add_argument('--configfiles', nargs='+', action=actions.CheckFile,
                        help='Configuration file names')
    parser.add_argument('--default_config', nargs=1, action=actions.CheckFile,
                        default=[default_config], dest='defconfig',
                        help='Default configuration file name')
    parser.add_argument('--cont', nargs='+', action=actions.CheckFile,
                        help='Continuum files')
    parser.set_defaults(data=None)

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    for step_name, step in steps.items():
        args.log.info('*' * 80)
        if step_name in args.skip:
            args.log.warning('Skipping step: %s', step_name)
            if step_name not in ['split']:
                args.log.info('*' * 80)
                continue
        else:
            args.log.info('Running step: %s', step_name)
        args.log.info('*' * 80)
        # Each step has to be run either way to store data directories and
        # other data needed in subsequent steps
        step(args)

if __name__ == '__main__':
    unic(sys.argv[1:])
