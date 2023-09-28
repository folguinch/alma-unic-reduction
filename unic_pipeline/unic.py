"""ALMA-UNIC pipeline."""
from typing import Optional, List
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
from pathlib import Path
import argparse
import sys

from casatasks import split, concat
from unic_pipepline import tasks
from unic_pipepline.data_handler import DataHandler
import unic_pipepline.argparse_parents as parents
import unic_pipepline.argparse_actions as actions

def _set_config(args: argparse.Namespace):
    """Setup the `config` value in `args`.

    It assumes:

    - A `configfile` arguments exists in the parser.
    - The initial value of `config` is a dictionary with default values for the
      `configparser`, or None.
    """
    # Read configuration
    default = args.config
    args.config = ConfigParser(interpolation=ExtendedInterpolation())
    if default is not None:
        args.log.info('Reading default configuration: %s', default)
        args.config.read(default)
    args.log.info('Reading input configuration: %s', args.configfile)
    args.config.read(args.configfile)

    # Update field
    if args.field is not None:
        args.config['DEFAULT']['field'] = args.field[0]

    # Update basedir
    if args.basedir is not None:
        args.config['DEFAULT']['basedir'] = f'{args.basedir}'

def split_vis(args: argparse.Namespace):
    """Split the visibilities for the requested source."""
    config = args.config['split']
    neb = len(args.uvdata)
    splitvis = []
    for i, vis in enumerate(args.uvdata):
        # This define the naming conventions for the splitted visibilities
        if neb > 1:
            outputvis = args.uvdata.parent / f"{config['name']}_eb{i+1}.ms"
        else:
            outputvis = args.uvdata.parent / f"{config['name']}.ms"

        run_step = tasks.validate_step('split' in args.skip, outputvis)
        if run_step:
            split(vis=vis,
                  outputvis=f'{outputvis}',
                  field=config['field'],
                  datacolumn=config['datacolumn'],
                  spw=config['spw'])
        splitvis.append(outputvis)

    # Concatenate if more than 1 EB
    if neb > 1:
        concatvis = args.uvdata.parent / f"{config['name']}.ms"
        run_step = tasks.validate_step('split' in args.skip, concatvis)
        if run_step:
            concat(list(map(str, splitvis)), concatvis=f'{concatvis}')
    else:
        concatvis = splitvis[0]

    # Generate a DataHandler
    args.data_handler = DataHandler(name=concatvis.stem, uvdata=concatvis,
                                    neb=neb)

def dirty_cubes(args: argparse.Namespace):
    """Calculate dirty cubes."""
    config = args.config['dirty_cubes']
    outdir = Path(config['directory'])
    outdir.mkdir(parents=True, exist_ok=True)
    args.data_handler.clean_per_spw(config,
                                    outdir,
                                    nproc=args.nproc[0],
                                    skip='dirty_cubes' in args.skip,
                                    log=args.log.info,
                                    niter=0)

def get_continuum(args: argparse.Namespace):
    """Obtain or create a line-free channel file."""
    config = args.config['get_cont']

    # Read or create file
    cont_file = Path(config.get('cont_file', fallback='./cont.txt'))
    if not cont_file.is_file():
        # Different methods can be inplemented in here, and the a "mode" or
        # "method" item can be added to the config file
        raise NotImplementedError('Continuum finding not implemented yet')
    args.data_handler.cont_file = cont_file

def contsub(args: argparse.Namespace):
    """Calculate the continuum subtracted visibilities.
    
    At the moment it is assumed that each line of the `cont.txt` contains the
    continuum channels for the corresponding `spw`. This is EB independent,
    i.e. lines do need to be repeated in the file.
    """
    config = args.config['contsub']
    args.data_handler.contsub(config,
                              skip='contsub' in args.skip,
                              log=args.log.info)

def cont_avg(args: argparse.Namespace):
    """Calculate the continuum visibilities."""
    config = args.config['continuum']
    args.data_handler.continuum(config,
                                skip='continuum' in args.skip,
                                log=args.log.info)

def clean_continuum(args: argparse.Namespace):
    """Clean continuum data."""
    config = args.config['clean_cont']
    outdir = Path(config['directory'])
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean data
    for suffix in ['cont_all', 'cont']:
        imagename = outdir / f'{args.data_handler.name}_{suffix}.image'
        uvdata = args.data_handler.uvdata.with_suffix(f'.ms.{suffix}')
        args.data_handler.clean_data(config,
                                     imagename,
                                     uvdata=uvdata,
                                     nproc=args.nproc[0],
                                     skip='clean_cont' in args.skip,
                                     log=args.log.info)

def clean_cubes(args: argparse.Namespace):
    """Clean cube data."""
    config = args.config['clean_cubes']
    outdir = Path(config['directory'])
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean data
    uvdata = args.data_handler.uvdata.with_suffix('.ms.contsub')
    args.data_handler.clean_per_spw(config,
                                    outdir,
                                    uvdata=uvdata,
                                    nproc=args.nproc[0],
                                    skip='clean_cubes' in args.skip,
                                    log=args.log.info)

def unic(args: Optional[List] = None) -> None:
    """Run the main UNIC pipeline."""
    # Pipeline steps and functions
    pipe = {'split': split_vis,
            'dirty_cubes': dirty_cubes,
            'get_cont': get_continuum,
            'contsub': contsub,
            'continuum': cont_avg,
            'clean_cont': clean_continuum,
            'clean_cubes': clean_cubes}

    # Default config
    default_config = (Path(__file__).resolve().parent /
                      'configs/default.cfg')

    # Collect all the steps
    steps = [_set_config]
    steps = steps + list(pipe.values())

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
    parser.add_argument('-b', '--basedir', default=None, nargs=1,
                        action=actions.NormalizePath,
                        help='Base directory')
    parser.add_argument('-n', '--nproc', nargs=1, type=int, default=[5],
                        help='Number of processes for parallel processing')
    parser.add_argument('--skip', nargs='*', choices=list(pipe.keys()),
                        default=[],
                        help='Steps to skip')
    parser.add_argument('--field', nargs=1, default=None,
                        help='Field name')
    parser.add_argument('configfile', nargs=1, action=actions.CheckFile,
                        help='Configuration file name')
    parser.add_argument('uvdata', nargs='*', type=str,
                        help='uv data ms')
    parser.set_defaults(config=default_config,
                        data_handler=None,)

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    for step_name, step in steps.items():
        if step_name in args.skip:
            args.log.warn('Skipping step: %s', step_name)
        else:
            args.log.info('Running step: %s', step_name)
        # Each step has to be run either way to store data directories and
        # other data needed in subsequent steps
        step(args)

if __name__ == '__main__':
    unic(sys.argv[1:])
