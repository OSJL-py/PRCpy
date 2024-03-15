# PRCpy: A Python Library for Physical Reservoir Computing

PRCpy is a Python package designed to ease experimental data processing for physical reservoir computing.

## Features

- Data handling and preprocessing.
- Customizable data processing pipelines for various research needs.

## Installation

PRCpy requires Python 3.9 or later.

#### Using pip

```bash
pip install prcpy
```

#### Using Poetry
```bash
poetry add prcpy
```

#### Note

Latest release is always recommended.
```bash
PIP: pip install prcpy --upgrade
POERTY: poetry update prcpy
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
mg_path = "mackey_glass_t17.npy"
target_values = get_npy_data(mg_path, norm=True)
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

## Authors & Maintainers

We are a [UCL Spintronics Group](https://www.ucl.ac.uk/spintronics/) at London Centre for Nanotechnology, University College London, UK. 
For any queries about PRCpy, please contact [Dr. Oscar Lee](zceesjl@ucl.ac.uk). 
For collaborations or research enquires, please contact [Prof. Hide Kurebayashi](https://www.ucl.ac.uk/spintronics/people/hidekazu-kurebayashi).

## Find out more on PRC

#### PRCpy
- Refer to [PRCpy documentation [TBA]]() for detailed package documentation.
- Refer to the [PRCpy tutorial paper]() for detailed tutorial use of PRCpy.

#### RC publications from the group

##### Research articles
- [Task-adaptive physical reservoir computing](https://www.nature.com/articles/s41563-023-01698-8), O. Lee, et al., Nature Materials (2024).
- [Neuromorphic Few-Shot Learning: Generalization in Multilayer Physical Neural Networks](https://arxiv.org/abs/2211.06373), K. Stenning, et al., arXiv (2024).
- [Reconfigurable training and reservoir computing in an artificial spin-vortex ice via spin-wave fingerprinting](https://www.nature.com/articles/s41565-022-01091-7), J. Gartside, et al., Nature Nanotechnology (2024).

##### Review/perspectives
- [Perspective on unconventional computing using magnetic skyrmions](https://pubs.aip.org/aip/apl/article/122/26/260501/2900466), O. Lee, et al., Applied Physics Letters (2023).
- [Memristive, Spintronic, and 2D-Materials-Based Devices to Improve and Complement Computing Hardware](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202200068), D. Joksas, et al., Advanced Intelligent Systems (2022).

##### Outreach
- [Physical reservoir computers that can adapt to perform different tasks](https://www.nature.com/articles/s41563-023-01708-9), H. Kurebayashi, O. Lee, Nature Materials Research Briefing (2024).



## Recent PRC publications
TBA.





