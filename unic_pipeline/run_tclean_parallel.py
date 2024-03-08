"""Script to run tclean in parallel with mpicasa."""
from pathlib import Path
import argparse
import json
import os
import sys

def main(args: list):
    """Run tclean."""
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action='store_true')
    parser.add_argument('uvdata', nargs=1, type=str)
    parser.add_argument('imagename', nargs=1, type=str)
    parser.add_argument('paramsfile', nargs=1, type=str)
    args = parser.parse_args(args)

    # Load params
    # pylint: disable=W1514
    paramsfile = Path(args.paramsfile[0])
    tclean_params = json.loads(paramsfile.read_text())

    # Save masks
    if tclean_params.get('usemask', '') == 'auto-multithresh':
        os.environ['SAVE_ALL_AUTOMASKS'] = 'true'

    # Run tclean
    # pylint: disable=E0602
    summary = tclean(vis=args.uvdata[0], imagename=args.imagename[0], **tclean_params)
    casalog.post('tclean summary: %s' % summary, 'INFO')

    # Unset variable
    if tclean_params.get('usemask', '') == 'auto-multithresh':
        os.environ['SAVE_ALL_AUTOMASKS'] = 'false'

if __name__ == '__main__':
    main(sys.argv[1:])
