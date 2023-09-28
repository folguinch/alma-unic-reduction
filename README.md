# ALMA-UNIC pipeline

## Installation

Before start, the installation will also install/update the dependencies. These
include modular CASA. If you have a system wide modular CASA installation that
you want to keep, then it is recommended to install the pipeline within a 
virtual environment.

To install the ALMA-UNIC pipeline:

```bash
pip install git+
```

For tasks requiring tclean, a monolithic CASA version is needed. If you prefer a
specific version of casa/mpicasa, then you can define the following environmental
variable:

```bash
export MPICASA="/path/to/mpicasa -n {0} /path/to/casa"
```

Note that only the paths to `casa` and `mpicasa` need to be modified, the number
of processes (after the `-n` flag) is evaluated internally from the command line
options.

### For developing

In general, to install in editable mode run:

```bash
# Clone to your desired directory
mkdir -p /my/preferred/directory
cd /my/preferred/directory
git clone ....

# Install with pip in editable mode
cd unic-pipeline
pip install -e .
```

If you use `poetry` to manage dependencies, then follow their instructions for
installation with the lock file.

All dependencies are managed in the `pyproject.toml` file within the base directory.

## Usage

For help:

```bash
python -m unic_pipeline.unic -h
```
