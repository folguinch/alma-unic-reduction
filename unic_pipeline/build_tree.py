#!/bin/python3
"""Build a directory tree and indetifies sources and arrays."""
from typing import Optional, List
from pathlib import Path
import argparse
import sys

from astropy.table import Table
from unic_pipeline.utils import get_targets, get_array
import unic_pipeline.argparse_actions as actions

def find_calibrated(directory: Path) -> List:
    """Find the calibrated MS within `directory`."""
    # Iterate over directory
    mss = []
    for ms in directory.glob('**/calibrated/*.ms'):
        if '_targets' in ms.name:
            continue
        mss.append(ms)

    return mss

def build_table(ms_list: List, filename: Path) -> None:
    """Build table with paths and sources."""
    table = Table(names=('field', 'array', 'ms'), dtype=(str, str, str))
    for ms in ms_list:
        # Get targets
        fields = get_targets(ms)

        # Get array
        array = get_array(ms)

        # Add rows
        for field in fields:
            table.add_row((field, array, f'{ms}'))

    table.sort(['field', 'array'])
    table.write(filename, format='ascii')

def build_tree(args: Optional[List] = None) -> None:
    """Search for MS within input directory."""
    # Command line options
    parser = argparse.ArgumentParser(
        description='Search for MS within input directory.',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('directory', nargs=1, action=actions.NormalizePath,
                        help='Base directory to iterate over')
    parser.add_argument('table', nargs=1, action=actions.NormalizePath,
                        help='Table filename to write results')

    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run through steps
    calibrated = find_calibrated(args.directory[0])
    build_table(calibrated, args.table[0])

if __name__ == '__main__':
    build_tree(sys.argv[1:])
