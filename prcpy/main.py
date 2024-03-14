import numpy as np
import matplotlib.pyplot as plt

from RC.Pipeline_RC import *
from TrainingModels.RegressionModels import *

if __name__ == "__main__":

    ## Quick Tutorial 

    ## loading data
    data_dir_path = "examples\data\mg_mapping\Cu2OSeO3\skyrmion"

    process_params = {
        "Xs": "Frequency",
        "Readouts": "Spectra",
        "remove_bg": True, 
        "bg_fname": "BG_450mT_1_0to6_0GHz_4K.txt", 
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

    rc_pipeline = Pipeline(data_dir_path, process_params)

    ## target_generation (transformation)
    period = 10
    sample_spacing = rc_pipeline.get_sample_spacing(period)
    # target_values = get_square_waves(sample_spacing, period, norm=True)
    target_values = get_mackey_glass(norm=True)
    rc_pipeline.define_target(target_values)

    ## define the training model. We will choose the Ridge regression.
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

    ## define the RC parameters
    rc_params = {
        "model": model,
        "tau": 10,
        "test_size": 0.3,
        "error_type": "MSE"
    }

    ## run the pipeline
    rc_pipeline.run(rc_params)

    ## get the results
    results = rc_pipeline.get_rc_results()

    ## plot results
    train_ts = np.arange(results["train"]["y_train"].shape[0])
    test_ts = np.arange(results["test"]["y_test"].shape[0])
    train_ys = results["train"]["y_train"]
    test_ys = results["test"]["y_test"]
    train_preds = results["train"]["train_pred"]
    test_preds = results["test"]["test_pred"]

    plt.figure()
    plt.plot(train_ts, train_ys, label="Train")
    plt.plot(train_ts, train_preds, label="Train predict")
    plt.figure()
    plt.plot(test_ts, test_ys, label="Test")
    plt.plot(test_ts, test_preds, label="Test predict")
    plt.legend()
    plt.show()


