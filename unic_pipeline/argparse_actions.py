"""Collection of actions to process diferent command line inputs."""
from typing import List, Union
import argparse
import pathlib

from .logger import get_stdout_logger, update_logger

def validate_path(path: pathlib.Path,
                  check_is_file: bool = False,
                  check_is_dir: bool = False,
                  mkdir: bool = False):
    """Performs several checks on input path.

    Args:
      path: path to check.
      check_is_file: optional; check whether is file and exists.
      check_is_dir: optional; check whther is a directory and exists.
      mkdir: optional; make directories if they do not exist.
    Returns:
      A validated resolved pathlib.Path object
    """
    path = path.expanduser().resolve()
    if check_is_file and not path.is_file():
        raise IOError(f'{path} does not exist')
    elif check_is_dir or mkdir:
        if not path.is_dir() and not mkdir:
            raise IOError(f'{path} directory does not exist')
        else:
            path.mkdir(parents=True, exist_ok=True)

    return path

def validate_paths(filenames: Union[str, List[str]],
                   check_is_file: bool = False,
                   check_is_dir: bool = False,
                   mkdir: bool = False):
    """Performs several checks on input list of file names.

    Args:
      filenames: list of filenames to check.
      check_is_file: optional; check whether is file and exists.
      check_is_dir: optional; check whther is a directory and exists.
      mkdir: optional; make directories if they do not exist.
    Returns:
      Validate path.Path from the input strings.
    """
    try:
        # Case single string file name
        validated = pathlib.Path(filenames)
        validated = validate_path(validated, check_is_file=check_is_file,
                                  check_is_dir=check_is_dir, mkdir=mkdir)
    except TypeError:
        # Case multiple file names
        validated = []
        for filename in filenames:
            aux = pathlib.Path(filename)
            aux = validate_path(aux, check_is_file=check_is_file,
                                check_is_dir=check_is_dir, mkdir=mkdir)
            validated.append(aux)

    return validated

class NormalizePath(argparse.Action):
    """Normalizes a path or filename."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = validate_paths(values)
        setattr(namespace, self.dest, values)

class CheckFile(argparse.Action):
    """Validates files and check if they exist."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = validate_paths(values, check_is_file=True)
        setattr(namespace, self.dest, values)

# Logger actions
class StartLogger(argparse.Action):
    """Create a logger.

    If nargs=? (default), log to default or const if provided and flag used
    else log only to stdout.
    For verbose levels, the standard option_string values are:
      -v, --vv, --vvv, --log, --info, --debug, --fulldebug
    With: -v = --log = --info
          --vv = --debug
          --vvv = --fulldebug
    Other values will create a normal logger.
    """

    def __init__(self, option_strings, dest, nargs='?', metavar='LOGFILE',
                 const=None, default=None, **kwargs):
        # Cases from nargs
        if nargs not in ['?']:
            raise ValueError('nargs value not allowed')

        # Default dest and default log file
        dest = 'log'
        self._logfile = const or default

        # Set default to stdout
        const = None
        default = get_stdout_logger('__main__', verbose='v')

        super().__init__(option_strings, dest, nargs=nargs, metavar=metavar,
                         const=const, default=default, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if value is None:
            value = self._logfile

        # Determine verbose
        if option_string in ['-v', '--log', '--info']:
            verbose = 'v'
        elif option_string in ['--vv', '--debug']:
            verbose = 'vv'
        elif option_string in ['--vvv', '--fulldebug']:
            verbose = 'vvv'
        else:
            verbose = None

        logger = update_logger(self.default, filename=value, verbose=verbose)
        setattr(namespace, self.dest, logger)

