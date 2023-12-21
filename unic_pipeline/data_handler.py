"""Handler for the data."""
from typing import Optional, List, Callable, Dict, Tuple, Sequence
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, field, InitVar
from pathlib import Path
import os

from casatasks import uvcontsub, split, concat

from .utils import (get_spws_indices, validate_step, get_targets, get_array,
                    find_spws)
from .clean_tasks import get_tclean_params, tclean_parallel
from .common_types import SectionProxy

def new_array_dict(arrays: Sequence = ('12m', '7m')):
    return {array: () for array in arrays}

@dataclass
class FieldHandler:
    """Keep track of field data."""
    name: Optional[str] = None
    """Name of the field."""
    array: Optional[str] = None
    """Array name."""
    config: Optional[ConfigParser] = field(default=None, init=False)
    """Configuration parser."""
    log: 'logging.Logger' = None
    """Logging object."""
    uvdata: Optional[Path] = None
    """Measurement set directory."""
    configfile: Optional[Path] = None
    """Configuration file name."""
    default_config: InitVar[Optional[Path]] = None
    """Default configuration file name."""
    datadir: InitVar[Path] = Path('./').resolve()
    """Data directory."""
    neb: Optional[int] = None
    """Number of execution blocks in the uvdata."""
    cont_file: Optional['pathlib.Path'] = None
    """Continuum file."""
    spws: List[str] = field(default_factory=list, init=False)
    """List with spw values."""

    def __post_init__(self, default_config, datadir):
        # Check log
        if self.log is None:
            raise NotImplementedError('A logging.Logger is required')

        # Check uvdata name
        if (self.uvdata is None and
            self.field is not None and
            self.array is not None):
            self.uvdata = datadir / f'{self.field}_{self.array}.ms'
            self.log.debug(f'Setting uvdata name as: %s', self.uvdata)

        # Open configfile
        if self.configfile is not None or default_config is not None:
            if self.configfile is None:
                self.configfile = self.uvdata.with_suffix('.cfg')
            self.read_config(default_config=default_config)
        self._validate_config()

        # Fill spws
        if self.uvdata is not None and self.uvdata.exists():
            self.spws = get_spws_indices(self.uvdata)

    @property
    def field(self):
        return self.name

    @field.setter
    def field(self, value: str):
        self.name = value
    
    def _validate_config(self):
        """Check that data in config and FieldHandler coincide."""
        # Check if config has been initiated
        if self.config is None:
            raise ValueError('Cannot create FieldHandler without config')

        # Field name
        if self.field is None:
            self.field = self.config['DEFAULT'].get('field')
            self.log.debug('Copying field from config: %s', self.field)
        elif self.field != self.config['DEFAULT']['field']:
            self.log.debug('Updating field in config to: %s', self.field)
            self.config['DEFAULT']['field'] = self.field
        
        # Check array
        if self.array is None:
            self.array = self.config['DEFAULT'].get('array')
            self.log.debug('Copying array from config: %s', self.array)
        elif self.array != self.config['DEFAULT']['array']:
            self.log.debug('Updating array in config to: %s', self.array)
            self.config['DEFAULT']['array'] = self.array

        # Check uvdata
        if self.uvdata is None:
            datadir = Path(self.config['DEFAULT']['basedir'])
            self.uvdata = datadir / f'{self.field}_{self.array}.ms'
            self.log.debug(f'Setting uvdata name as: %s', self.uvdata)
        else:
            basedir1 = self.uvdata.parent
            basedir2 = Path(self.config['DEFAULT']['basedir'])
            if basedir1 != basedir2:
                self.log.debug('Updating datadir in config to: %s', basedir1)
                self.config['DEFAULT']['basedir'] = f'{basedir1}'
            else:
                pass

        # Check number of EBs
        if self.neb is None:
            self.neb = self.config['DEFAULT'].getint('neb')
            self.log.debug('Copying number of EBs from config: %s', self.neb)
        elif self.neb != self.config['DEFAULT'].getint('neb'):
            self.log.debug('Updating number of EBs in config to: %s', self.neb)
            self.config['DEFAULT']['neb'] = f'{self.neb}'

    def read_config(self,
                    default_config: Optional['pathlib.Path'] = None) -> None:
        """Load a configuration file."""
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        if default_config is not None and default_config.is_file():
            self.log.info(f'Reading default configuration: {default_config}')
            self.config.read(default_config)
        if self.configfile is not None and self.configfile.is_file():
            self.log.info(f'Reading input configuration: {self.configfile}')
            self.config.read(self.configfile)
        elif self.configfile is not None and not self.configfile.is_file():
            self.update_config(write=True,
                               DEFAULT={'field': self.field,
                                        'array': self.array,
                                        'basedir': f'{self.uvdata.parent}',
                                        'neb': f'{self.neb}'})

    def update_config(self, write: bool = False, **sections):
        """Update and write the configuration file."""
        # Update file
        self.config.read_dict(sections)

        # Write
        if write:
            self.write_config()

    def write_config(self):
        """Write config file."""
        # Write file
        self.log.info(f'Writing configuration: {self.configfile}')
        with self.configfile.open('w') as configfile:
            self.config.write(configfile)

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

@dataclass
class FieldManager:
    """Manage field information."""
    log: 'logging.Logger'
    """Logging object."""
    resume: bool = False
    """Resume or delete already created data?"""
    original: Dict[str, Tuple[Path]] = field(default_factory=new_array_dict)
    """Original data for each array."""
    split: Dict[str, Tuple[Path]] = field(default_factory=new_array_dict)
    """Split data for each array."""
    data_handler: Dict[str, FieldHandler] = field(default_factory=dict)
    """Data handler."""
    field: InitVar[Optional[str]] = None
    """Field name."""
    original_uvdata: InitVar[Optional[Path]] = None
    """Original data."""
    datadir: InitVar[Path] = Path('./')
    """Data directory."""
    default_config: InitVar[Optional[Path]] = None
    """Default configuration file name."""
    configfile: InitVar[Optional[Path]] = None
    """Configuration file name."""

    def __post_init__(self, field, original_uvdata, datadir, default_config,
                      configfile):
        if original_uvdata is not None:
            self.append_data(field=field,
                             original_uvdata=original_uvdata,
                             default_config=default_config,
                             configfile=configfile,
                             datadir=datadir)

    @property
    def name(self):
        return self.data_handler[list(self.data_handler.keys())[0]].field

    def datadir(self, array: str) -> Path:
        return self.data_handler[array].uvdata.parent

    def _init_data_handler(self,
                           field: str,
                           array: str,
                           **kwargs):
        self.data_handler[array] = FieldHandler(field,
                                                array,
                                                self.log,
                                                **kwargs)

    def append_data(self,
                    field: Optional[str] = None,
                    original_uvdata: Optional[Path] = None,
                    split_uvdata: Optional[Path] = None,
                    default_config: Optional[Path] = None,
                    configfile: Optional[Path] = None,
                    datadir: Optional[Path] = None):
        """Update information in field."""
        if original_uvdata is not None:
            # Store original data
            array = get_array(original_uvdata)
            self.log.debug('Data array: %s', array)
            self.original[array] += (original_uvdata,)
            self.log.debug('Original data: %s', self.original)

            # Update handler
            neb = len(self.original[array])
            if (handler:= self.data_handler.get(array)) is None:
                self.log.debug('Generating new data handler')
                self._init_data_handler(field,
                                        array,
                                        default_config=default_config,
                                        configfile=configfile,
                                        datadir=datadir,
                                        neb=neb)
            else:
                self.log.debug('Updating data handler')
                handler.neb = neb
                handler.update_config(DEFAULT={'neb': f'{neb}'})

            # Split?
            split = self.datadir(array) / f'{self.name}_{array}_eb{neb}.ms'
            if split.exists() and not self.resume:
                self.log.warning('Deleting split ms: %s', split)
                os.system(f'rm -rf {split}')
            if not split.exists():
                self.log.info('Splitting input ms to: %s', split)
                self.split_field(array, original_uvdata, split)
            else:
                self.log.info('Found split ms: %s', split)
            self.split[array] += (split,)
            self.log.debug('Split data: %s', self.split)

    def split_field(self, array: str, uvdata: Path, splitvis: Path):
        """Split field from input uvdata."""
        # Split parameters
        if ((spw := self.data_handler[array].config.get(
            'split', 'spw', fallback=None)) is None):
            spw = find_spws(self.name, uvdata)
            self.data_handler[array].update_config(split={'spw': spw})
        datacolumn = self.data_handler[array].config['split']['datacolumn']

        # Perform splitting
        self.log.info('Selected spws for split: %s', spw)
        self.log.info('Selected datacolumn for split: %s', datacolumn)
        split(vis=f'{uvdata}', outputvis=f'{splitvis}', field=self.name,
              datacolumn=datacolumn, spw=spw)

    def concat_data(self):
        """Concatenate stored data per array."""
        for array, handler in self.data_handler.items():
            # Prepare concat input
            vis = [f'{val}' for val in self.split[array]]
            concatvis = handler.uvdata

            # Check data in disk
            if concatvis.exists() and not self.resume:
                self.log.warning('Deleting concat ms: %s', concatvis)
                os.system(f'rm -rf {concatvis}')
            elif concatvis.exists() and self.resume:
                continue
            
            # Concat
            if len(vis) == 0:
                args.log.warning('Data handler created but no data: %s', array)
                raise ValueError('Missing data')
            elif len(vis) == 1:
                self.log.info('Moving %s to: %s', vis[0], concatvis)
                os.system(f'mv {vis[0]} {concatvis}')
            else:
                self.log.info('MSs to concat: %s', vis)
                self.log.info('Concatenating MSs to: %s', concatvis)
                concat(vis=vis, concatvis=f'{concatvis}')

    def write_configs(self):
        """Write configs to disk."""
        for array, handler in self.data_handler.items():
            handler.write_config()

class DataManager(Dict):
    """Manage pipeline data."""
    def __init__(self, log: 'logging.Logger', **kwargs):
        self.log = log
        super().__init__(kwargs)

    @classmethod
    def from_uvdata(cls,
                    uvdata: Path,
                    default_config: Path,
                    log: 'logging.Logger',
                    datadir: Path = Path('./'),
                    resume: bool = False):
        """Generate a `DataManager` from input MSs.

        It uses the input uvdata to find target fields and generates a
        configuration file for each.
        """
        # First pass: iterate over uvdata and define fields
        fields = {}
        for ms in uvdata:
            print('=' * 80)
            log.info('Operating over MS: %s', ms)
            targets = get_targets(ms)
            log.info('Fields in MS: %s', targets)

            # Iterate over fields
            for field in targets:
                print('-' * 80)
                if field in fields:
                    log.info('Updating field: %s', field)
                    fields[field].append_data(original_uvdata=ms) 
                    continue
                # Read config
                log.info('Setting up field: %s', field)
                field_info = FieldManager(log, resume=resume, field=field,
                                          original_uvdata=ms,
                                          default_config=default_config,
                                          datadir=datadir)
                fields[field] = field_info

        # Second pass: iterate over fields to concatenate data and save config
        # files
        for name, field in fields.items():
            print('=' * 80)
            log.info('Concatenating field: %s', name)
            field.concat_data()
            log.info('Writing configs')
            field.write_configs()

        return cls(log=log, **fields)

    @classmethod
    def from_configs(cls,
                     configfiles: Path,
                     log: 'logging.Logger',
                     datadir: Path = Path('./'),
                     resume: bool = False):
        """Generate a `DataManager` from input configuration files.

        At the moment it assumes the data is ready for processesing.
        """
        # Iterate over files
        fields = {}
        for cfg in configfiles:
            # Create field handler
            handler = FieldHandler(configfile=cfg, datadir=datadir, log=log)

            # Update field
            if (field := handler.field) in fields:
                if (array := handler.array) in fields[field].data_handler:
                    raise KeyError(
                        'Loading 2 configs for the same field and array')
                fields[field].data_handler[array] = handler
            else:
                fields[field] = FieldManager(
                    data_handler={handler.array: handler},
                    log=log,
                    resume=resume)

        return cls(log=log, **fields)
