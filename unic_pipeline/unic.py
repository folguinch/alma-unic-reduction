"""ALMA-UNIC pipeline."""
from typing import Optional, List
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
from pathlib import Path
import argparse
import sys

from casatasks import split, concat
from unic_pipeline import utils
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
            resume='split' in args.skip or args.resume,
        )
    elif args.uvdata is not None:
        args.data = DataManager.from_uvdata(
            args.uvdata,
            args.defconfig,
            args.log,
            datadir=args.basedir[0],
            resume='split' in args.skip or args.resume,
        )
    else:
        raise ValueError('Need input data')

#def _set_uvdata(args: argparse.Namespace):
#    """Setup the uvdata value.
#
#    If `args.uvdata` is not set and `field.ms` is in the directory, then
#    `split` and `concat` steps are set to skip them. In this case, number of EBs
#    should be set in the configuration or command line if more than 1.
#    """
#    if args.uvdata is None:
#        # Set MS
#        name = args.config['split']['name']
#        args.log.info('Looking for %s MSs in the directory %s', name,
#                      args.basedir)
#        concatvis = args.basedir / f'{name}.ms'
#
#        # Look for number of EBs
#        if args.neb is None:
#            args.neb = [args.config.getint('split', 'neb', fallback=1)]
#
#        # Set uvdata and skip values
#        if concatvis.exists():
#            args.uvdata = [concatvis] * args.neb
#            args.skip = args.skip + ['split']
#        else:
#            raise ValueError('Could not find source MS')
#    else:
#        args.neb = [len(args.uvdata)]

#def split_vis(args: argparse.Namespace):
#    """Split the visibilities for the requested source."""
#    config = args.config['split']
#    splitvis = []
#    args.log.debug('Number of EBs: %i', args.neb[0])
#    for i, vis in enumerate(args.uvdata):
#        # This define the naming conventions for the splitted visibilities
#        if args.neb[0] > 1:
#            outputvis = args.basedir / f"{config['name']}_eb{i+1}.ms"
#        else:
#            outputvis = args.basedir / f"{config['name']}.ms"
#
#        run_step = utils.validate_step('split' in args.skip, outputvis,
#                                       log=args.log.warning)
#        if run_step:
#            args.log.info('Splitting MS: %s', vis)
#            args.log.info('Output MS: %s', outputvis)
#            args.log.info('Using column: %s', config['datacolumn'])
#            split(vis=vis,
#                  outputvis=f'{outputvis}',
#                  field=config['field'],
#                  datacolumn=config['datacolumn'],
#                  spw=config['spw'])
#        splitvis.append(outputvis)
#
#    # Concatenate if more than 1 EB
#    if args.neb[0] > 1:
#        concatvis = args.basedir / f"{config['name']}.ms"
#        run_step = utils.validate_step('split' in args.skip, concatvis,
#                                       log=args.log.warning)
#        if run_step:
#            args.log.info('Creating concat MS: %s', concatvis)
#            concat(list(map(str, splitvis)), concatvis=f'{concatvis}')
#    else:
#        concatvis = splitvis[0]
#
#    # Generate a DataHandler
#    args.data_handler = DataHandler(name=concatvis.stem, uvdata=concatvis,
#                                    neb=args.neb[0])
#
#    print(args.data_handler)
#    raise Exception
#
#def dirty_cubes(args: argparse.Namespace):
#    """Calculate dirty cubes."""
#    config = args.config['dirty_cubes']
#    outdir = Path(config['directory'])
#    outdir.mkdir(parents=True, exist_ok=True)
#    args.data_handler.clean_per_spw(config,
#                                    outdir,
#                                    nproc=args.nproc[0],
#                                    skip='dirty_cubes' in args.skip,
#                                    log=args.log.info,
#                                    niter=0)
#
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
#def contsub(args: argparse.Namespace):
#    """Calculate the continuum subtracted visibilities.
#    
#    At the moment it is assumed that each line of the `cont.txt` contains the
#    continuum channels for the corresponding `spw`. This is EB independent,
#    i.e. lines do need to be repeated in the file.
#    """
#    config = args.config['contsub']
#    args.data_handler.contsub(config,
#                              skip='contsub' in args.skip,
#                              log=args.log.info)
#
#def cont_avg(args: argparse.Namespace):
#    """Calculate the continuum visibilities."""
#    config = args.config['continuum']
#    args.data_handler.continuum(config,
#                                skip='continuum' in args.skip,
#                                log=args.log.info)
#
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
    steps = {#'split': split_vis,
             'split': prep_data,}
             #'dirty_cubes': dirty_cubes,
             #'get_cont': get_continuum,
             #'contsub': contsub,
             #'continuum': cont_avg,
             #'clean_cont': clean_continuum,
             #'clean_cubes': clean_cubes}

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
