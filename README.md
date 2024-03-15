# PRCpy: A Python Library for Physical Reservoir Computing

PRCpy is a Python package designed to ease experimental data processing for physical reservoir computing.

## Features

- Data handling and preprocessing.
- Customizable data processing pipelines for various research needs.

## Installation

PRCpy requires Python 3.9 or later.

### Using pip

```bash
pip install prcpy
```

### Using Poetry
```bash
poetry add prcpy
```

## General usage overview

1. Define data path 
2. Define pre-processing parameters 
3. Create RC pipeline
4. Define target and add to pipeline
5. Define model for training
6. Define RC parameters
7. Run RC

## Example:

#### import PRCpy
```python
from prcpy.RC import Pipeline
from prcpy.TrainingModels.RegressionModels import define_Ridge
from prcpy.Maths.Target_functions import get_mackey_glass, get_square_waves
```

#### Define data directory and processing parameters
**Note: Data files must contain _"scan"_ in their file names.**
See [examples/data](examples/data) for example data files.
```python
data_dir_path = "your/data/path"
process_params = {
    "Xs": "Frequency",
    "Readouts": "Spectra",
    "remove_bg": True,
    "bg_fname": "background_data.txt",
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

#### Create RC pipeline
```python
rc_pipeline = Pipeline(data_dir_path, process_params)
```

#### Target generation

##### Transformation
```python

period = 10
sample_spacing = rc_pipeline.get_sample_spacing(period)
target_values = get_square_waves(sample_spacing, period, norm=True)
```

##### Forecasting
```python
target_values = get_mackey_glass(norm=True)
```

##### Add target to pipeline
```python
rc_pipeline.define_target(target_values)
```

#### Define model
```python
model_params = {
        "alpha": 1e-3,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.0001,
        "solver": "auto",
        "positive": False,
        "random_state": None,
    }
model = define_Ridge(model_params)
```

#### Define RC parameters
Set `"tau": 0` for transformation.

```python
rc_params = {
        "model": model,
        "tau": 10,
        "test_size": 0.3,
        "error_type": "MSE"
    }

```

#### Run RC
```python
rc_pipeline.run()
```

#### Get results
```python
results = rc_pipeline.get_rc_results()
```

## Contributing

Any community contributions are welcome. Please refer to the project's GitHub repository for contribution guidelines.

## Authors

- Oscar Lee <zceesjl@ucl.ac.uk>


