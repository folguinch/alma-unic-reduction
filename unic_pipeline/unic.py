"""ALMA-UNIC pipeline."""
from typing import Optional, List
from configparser import ConfigParser, NoSectionError, ExtendedInterpolation
from datetime import datetime
from pathlib import Path
import argparse
import sys

from casatasks import split, concat
import unic_pipepline.argparse_parents as parents
import unic_pipepline.argparse_actions as actions
import unic_pipepline.tasks as tasks

def validate_step(in_skip: bool, filename: Path) -> bool:
    """Validate the step.

    If the step is not in skip and `filename` exists, then it is deleted and
    return `True` to run the step again. If the step is in skip but `filename`
    does not exist, then run the step to generate it.
    """
    if in_skip and filename.exists():
        return False
    elif filename.exists():
        os.system(f'rm -rf {filename}')

    return True

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
    args.log.info('Reading input configuration: %s', args.config)
    args.config.read(args.configfile)

    # Check field
    if args.field is not None:
        args.config['DEFAULT']['field'] = args.field[0]

    # Check basedir
    if args.basedir is not None:
        args.config['DEFAULT']['basedir'] = f'{args.basedir}'

def split_vis(args: argparse.Namespace):
    """Split the visibilities for the requested source."""
    config = args.config['split']
    neb = len(args.uvdata)
    splitvis = []
    for i, vis in enumerate(args.uvdata):
        if neb > 1:
            outputvis = args.uvdata.parent / f"{config['name']}_eb{i+1}.ms"
        else:
            outputvis = args.uvdata.parent / f"{config['name']}.ms"

        run_step = validate_step('split' in args.skip, outputvis)
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
        run_step = validate_step('split' in args.skip, concatvis)
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
    args.data_handler.clean_cubes(config,
                                  outdir,
                                  nproc=args.nproc[0],
                                  skip='dirty_cubes' in args.skip,
                                  log=args.log.info,
                                  niter=0)

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
    steps = step + list(steps.values())

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
        args.log.info('Running step: %s', step_name)
        step(args)

if __name__ == '__main__':
    unic(sys.argv[1:])
