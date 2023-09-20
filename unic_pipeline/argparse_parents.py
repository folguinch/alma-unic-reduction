"""Argparse parent parsers commonly used."""
from typing import Optional
import argparse
import pathlib

from casatasks import casalog

from .argparse_actions import StartLogger

def logger(filename: Optional['pathlib.Path'] = None,
           use_casa: bool = False) -> argparse.ArgumentParser:
    """Parent parser to initiate a logging system.

    Args:
      filename: optional; default filename for logging.
    """
    parser = argparse.ArgumentParser(add_help=False)
    if use_casa:
        parser.add_argument('--logfile', default=None, nargs=1, type=str,
                            help='Log file name')
        parser.set_defaults(log=casalog)
    else:
        parser.add_argument('-v', '--vv', '--vvv', default=filename,
                            action=StartLogger,
                            help='Logging setup')

    return parser
