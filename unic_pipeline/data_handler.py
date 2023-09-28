"""Handler for the data."""
from typing import Optional, List, Callable, Mapping
from dataclasses import dataclass, field
import os

from casatasks import uvcontsub, split

from .utils import get_spws_indices, validate_step
from .clean_tasks import get_tclean_params, tclean_parallel

SectionProxy = Mapping[str, str]

@dataclass
class DataHandler:
    """Keep track of data in UNIC pipeline."""
    name: Optional[str] = None
    """Name of the data."""
    uvdata: Optional['pathlib.Path'] = None
    """Measurement set directory."""
    neb: Optional[int] = None
    """Number of execution blocks in the uvdata."""
    cont_file: Optional['pathlib.Path'] = None
    """Continuum file."""
    spws: List[str] = field(default_factory=list, init=False)
    """List with spw values."""

    def __post_init__(self):
        # Fill spws
        if self.uvdata is not None:
            self.spws = get_spws_indices(self.uvdata)

    def map_cont_file(self) -> str:
        """Read `cont_file` and assign its lines to theirs `spws`."""
        lines = self.cont_file.read_text().split('\n')
        mapped = {}
        for spws, line in zip(self.spws, lines):
            mapped.update({spw: line for spw in spws.split(',')})
        mapped = ','.join(map(lambda x: f'{x[0]}:{x[1]}',
                              sorted(mapped.items())))

        return mapped

    def clean_data(self,
                   config: SectionProxy,
                   imagename: 'pathlib.Path',
                   uvdata: Optional['pathlib.Path'] = None,
                   nproc: int = 5,
                   skip: bool = False,
                   log: Callable = print,
                   **tclean_args):
        """Run tclean on input or stored uvdata.

        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          config: configuration proxy.
          imagename: image directory path.
          uvdata: optional; uvdata path.
          nproc: optional; number of processors for parallel clean.
          skip: optional; skip this step if files were created?
          log: optional; logging function.
          tclean_args: optional; additional arguments for tclean.
        """
        # tclean args
        kwargs = get_tclean_params(config, cfgvars=tclean_args)
        if uvdata is None:
            uvdata = self.uvdata

        # Check for files
        if imagename.exists() and skip:
            return
        elif imagename.exists() and not skip:
            os.system(f"rm -rf {imagename.with_suffix('.*')}")

        # Run tclean
        tclean_parallel(uvdata, imagename.with_name(imagename.stem),
                        nproc, kwargs, log=log)

    def clean_per_spw(self,
                      config: SectionProxy,
                      path: 'pathlib.Path',
                      uvdata: Optional['pathlib.Path'] = None,
                      nproc: int = 5,
                      skip: bool = False,
                      log: Callable = print,
                      **tclean_args):
        """Run `tclean` on all spws.
        
        Arguments given under `tclean_args` replace the values from `config`.

        Args:
          config: configuration proxy.
          path: output directory path.
          uvdata: optional; uvdata path.
          nproc: optional; number of processors for parallel clean.
          skip: optional; skip this step if files were created?
          log: optional; logging function.
          tclean_args: optional; additional arguments for tclean.
        """
        for i, spw in enumerate(self.spws):
            # Clean args
            imagename = path / f'{self.name}_spw{i}.image'
            if uvdata is None:
                uvdata = self.uvdata

            # Check for files
            if imagename.exists() and skip:
                log(f'Skipping cube for spw {i}')
                continue
            elif imagename.exists() and not skip:
                os.system(f"rm -rf {imagename.with_suffix('.*')}")

            # Run clean
            self.clean_data(config, imagename, uvdata=uvdata, nproc=nproc,
                            skip=skip, log=log, spw=spw, **tclean_args)

    def contsub(self,
                config: SectionProxy,
                skip: bool = False,
                log: Callable = print) -> None:
        """Run `uvcontsub` to obtain continuum subtracted visibilities."""
        outputvis = self.uvdata.with_suffix('.ms.contsub')
        val_step = validate_step(skip, outputvis)
        if not val_step:
            return

        # Prepare input
        fitspec = self.map_cont_file()
        fitorder = config.getint('fitorder')

        # Run uvcontsub
        uvcontsub(vis=f'{self.uvdata}', outputvis=f'{outputvis}',
                  fitspec=fitspec, fitorder=fitorder)

    def continuum(self,
                  config: SectionProxy,
                  skip: bool = False,
                  log: Callable = print) -> None:
        """Generate continuum visibilities.

        It generates 2 visibility sets:

        - Average of all the channels using the input width.
        - Average of the line free channels using the `cont_file` and width.
        """
        # Check files
        allchans = self.uvdata.with_suffix('.ms.cont_all')
        linefree = self.uvdata.with_suffix('.ms.cont')
        val_all = validate_step(skip, allchans)
        val_lf = validate_step(skip, linefree)

        # Values from config
        width = config['width']
        datacolumn = config['datacolumn']

        if val_all:
            # Average channels without flagging
            split(vis=f'{self.uvdata}', outputvis=f'{allchans}', width=width,
                  datacolumn=datacolumn)
        if val_lf:
            # Get line free channels
            cont_chans = self.map_cont_file()

            # Split the uvdata
            # WARNING: this needs to be tested
            # otherwise, use flagmanager and then split the visibilities
            split(vis=f'{self.uvdata}', outputvis=f'{linefree}',
                  spw=cont_chans, width=width)

