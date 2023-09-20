"""Handler for the data."""
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field, InitVar

from .utils import get_spws_indices

@dataclass
class DataHandler:
    """Keep track of data in UNIC pipeline."""
    name: Optional[str] = None
    """Name of the data."""
    uvdata: Optional['pathlib.Path'] = None
    """Measurement set directory."""
    neb: Optional[int] = None
    """Number of execution blocks in the uvdata."""
    spws: List[str] = field(default_factory=list, init=False)
    """Dictionary with spw values."""

    def __post_init__(self):
        # Fill spws
        if self.uvdata is not None:
            slef.spws = get_spws_indices(self.uvdata)

    def clean_cubes(self,
                    config: SectionProxy,
                    path: 'pathlib.Path',
                    nproc: int = 5,
                    skip: bool = False,
                    log: Callable = print,
                    **tclean_args):
        for i, spw in enumerate(self.spws):
            # tclean args
            kwargs = get_tclean_params(config, cfgvars=tclean_args)

            # Other args
            imagename = path / f'{name}_spw{i}.image'
            kwargs['spw'] = spw

            # Check for files
            if imagename.exists() and skip:
                log(f'Skipping cube for spw {i}')
                continue
            elif imagename.exists() and not skip:
                os.system(f"rm -rf {imagename.with_suffix('.*')}")

            # Run tclean
            tclean_parallel(self.uvdata, imagename.with_name(imagename.stem),
                            nproc, kwargs, log=log)

