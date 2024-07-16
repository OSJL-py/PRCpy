import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from prcpy.RC.Pipeline_RC import *
from prcpy.TrainingModels.RegressionModels import *
from prcpy.Maths.Target_functions import get_npy_data

if __name__ == "__main__":

    # Loading data
    data_dir_path = "examples\data\mg_mapping\capacitor_diode"
    prefix = "scan"

    process_params = {
        "Xs": "t",
        "Readouts": "Voltage",
        "remove_bg": False, 
        "smooth": False, 
        "smooth_win": 51, 
        "smooth_rank": 4, 
        "cut_xs": False, 
        "x1": 2, 
        "x2": 5, 
        "normalize": False, 
        "sample": False, 
        "sample_rate": 13,
        "delimiter": '\t',
        "transpose":True
    }

    rc_pipeline = Pipeline(data_dir_path,prefix,process_params)

    # Mackey Glass target generation (prediction)
    mg_path = "examples/data/chaos/mackey_glass_t17.npy"
    target_values = get_npy_data(mg_path, norm=True)
    rc_pipeline.define_target(target_values)

    # Define model parameters
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

    # Define the training model
    model = define_Ridge(model_params)

    # Define the RC parameters
    rc_params = {
        "model": model,
        "tau": 10,
        "test_size": 0.3,
        "error_type": "MSE"
    }

    # Run the pipeline
    rc_pipeline.run(rc_params)

    # Get the results
    results = rc_pipeline.get_rc_results()

    # Results
    train_ts = np.arange(results["train"]["y_train"].shape[0])
    test_ts = np.arange(results["test"]["y_test"].shape[0])
    train_ys = results["train"]["y_train"]
    test_ys = results["test"]["y_test"]
    train_preds = results["train"]["train_pred"]
    test_preds = results["test"]["test_pred"]

    # Errors
    train_MSE = results["error"]["train_error"]
    test_MSE = results["error"]["test_error"]

    print(f"Training MSE = {format(train_MSE, '0.3e')}")
    print(f"Testing MSE = {format(test_MSE, '0.3e')}")

    # Plot results
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{}".format(os.path.basename(data_dir_path)),size=20)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(train_ts,train_ys,label="Train")
    ax1.plot(train_ts,train_preds,label="Train predict")
    ax1.set_title(f"Training MSE: {format(train_MSE, '0.3e')}")
    ax1.legend()

    ax2.plot(test_ts, test_ys, label="Test")
    ax2.plot(test_ts, test_preds, label="Test predict")
    ax2.set_title(f"Testing MSE: {format(test_MSE, '0.3e')}")
    ax2.legend()

    fig.tight_layout()

    plt.show()


