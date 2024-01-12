"""ALMA-UNIC pipeline."""
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
            args.defconfig,
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
        data.dirty_cubes(nproc=args.nproc[0])

#def get_continuum(args: argparse.Namespace):
#    """Obtain or create a line-free channel file."""
#    config = args.config['get_cont']
#
#    # Read or create file
#    cont_file = Path(config.get('cont_file', fallback='./cont.txt'))
#    if not cont_file.is_file():
#        # Different methods can be inplemented in here, and the a "mode" or
#        # "method" item can be added to the config file
#        raise NotImplementedError('Continuum finding not implemented yet')
#    args.data_handler.cont_file = cont_file
#
def continuum(args: argparse.Namespace):
    """Calculate the continuum visibilities."""
    for data in args.data.values():
        data.continuum_visibilities(control_image=True, nproc=args.nproc[0])

def contsub(args: argparse.Namespace):
    """Calculate the continuum subtracted visibilities.

    At the moment it is assumed that each line of the `cont.txt` contains the
    continuum channels for the corresponding `spw`. This is EB independent,
    i.e. lines do need to be repeated in the file.
    """
    for data in args.data.values():
        data.contsub_visibilities(dirty_images=True, nproc=args.nproc[0])
#    config = args.config['contsub']
#    args.data_handler.contsub(config,
#                              skip='contsub' in args.skip,
#                              log=args.log.info)

#def clean_continuum(args: argparse.Namespace):
#    """Clean continuum data."""
#    config = args.config['clean_cont']
#    outdir = Path(config['directory'])
#    outdir.mkdir(parents=True, exist_ok=True)
#
#    # Clean data
#    for suffix in ['cont_all', 'cont']:
#        imagename = outdir / f'{args.data_handler.name}_{suffix}.image'
#        uvdata = args.data_handler.uvdata.with_suffix(f'.ms.{suffix}')
#        args.data_handler.clean_data(config,
#                                     imagename,
#                                     uvdata=uvdata,
#                                     nproc=args.nproc[0],
#                                     skip='clean_cont' in args.skip,
#                                     log=args.log.info)
#
#def clean_cubes(args: argparse.Namespace):
#    """Clean cube data."""
#    config = args.config['clean_cubes']
#    outdir = Path(config['directory'])
#    outdir.mkdir(parents=True, exist_ok=True)
#
#    # Clean data
#    uvdata = args.data_handler.uvdata.with_suffix('.ms.contsub')
#    args.data_handler.clean_per_spw(config,
#                                    outdir,
#                                    uvdata=uvdata,
#                                    nproc=args.nproc[0],
#                                    skip='clean_cubes' in args.skip,
#                                    log=args.log.info)

def unic(args: Optional[List] = None) -> None:
    """Run the main UNIC pipeline."""
    # Pipeline steps and functions
    steps = {
        'split': prep_data,
        'dirty_cubes': dirty_cubes,
        #'get_cont': get_continuum,
        'continuum': continuum,
        'contsub': contsub,
        #'clean_cont': clean_continuum,
        #'clean_cubes': clean_cubes,
    }

    # Default config
    default_config = (Path(__file__).resolve().parent /
                      'configs/default.cfg')

    # Collect all the steps
    #steps = dict(generate_configs=_generate_configs,
    #             set_configs=_set_configs,
    #             **pipe)

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
    parser.add_argument('--field', nargs=1, default=None,
                        help='Field name')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already created data.')
    parser.add_argument('--uvdata', nargs='+', type=str,
                        action=actions.NormalizePath,
                        help='uv data ms')
    parser.add_argument('--configfiles', nargs='+', action=actions.CheckFile,
                        help='Configuration file names')
    parser.add_argument('--cont', nargs='+', action=actions.CheckFile,
                        help='Continuum files')
    parser.set_defaults(data=None,
                        defconfig=default_config)

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    for step_name, step in steps.items():
        if step_name in args.skip:
            args.log.warning('Skipping step: %s', step_name)
        else:
            args.log.info('Running step: %s', step_name)
        # Each step has to be run either way to store data directories and
        # other data needed in subsequent steps
        step(args)

if __name__ == '__main__':
    unic(sys.argv[1:])
