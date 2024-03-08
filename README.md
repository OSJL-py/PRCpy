# PRCpy: A Python Library for Physical Reservoir Computing

PRCpy is a Python package designed to ease experimental data processing for physical reservoir computing.

## Features

- Data handling and preprocessing.
- Customizable data processing pipelines for various research needs.

## Installation

PRCpy requires Python 3.9 or later. You can install PRCpy using Poetry by adding it to your project's dependencies:


## Quick Start

Here's a quick example to get you started with PRCpy:

## import PRCpy
```python
from prcpy.RC.Pipeline_RC import 
from prcpy.TrainingModels.RegressionModels import
```

## Define your data directory and processing parameters
```python
data_dir_path = "examples/data/mg_mapping/Cu2OSeO3/skyrmion"
process_params = {
    "Xs": "Frequency",
    "Readouts": "Spectra",
    "remove_bg": True,
    "bg_fname": "BG_450mT_10to60GHz_4K.txt",
    "smooth": False,
    "smooth_win": 51,
    "smooth_rank": 4,
    "cut_xs": False,
    "x1": 2,
    "x2": 5,
    "normalize": False,
    "sample": True,
    "sample_rate": 13
}
```

# Create RC pipeline
rc = Pipeline_RC(data_dir_path, process_params)

# Get results


## Contributing

Any community contributions are welcome. Please refer to the project's GitHub repository for contribution guidelines.

## Authors

- Oscar Lee <zceesjl@ucl.ac.uk>


