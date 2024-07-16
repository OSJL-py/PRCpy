import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from prcpy.RC.Pipeline_RC import *
from prcpy.TrainingModels.RegressionModels import *
from prcpy.Maths.Target_functions import get_sawtooth_waves
from prcpy.DataHandling.Path_handlers import *

if __name__ == "__main__":

    data_dir_path = "examples\data\sine_mapping\Cu2OSeO3\conical_500"
    prefix = "scan"

    process_params = {
    "Xs": "Frequency",
    "Readouts":"Spectra",
    "remove_bg": True,
    "bg_fname": "BG_450mT_1_0to6_0GHz_4K.txt",
    "delimiter": ",",
    "normalize": True,
    "transpose": False,
    }

    rc_pipeline = Pipeline(data_dir_path,prefix,process_params)

    reservoir_df = rc_pipeline.rc_data.rc_df
    #print(reservoir_df)

    # Square wave target generation (transformation)
    period = 10
    sample_spacing = rc_pipeline.get_sample_spacing(period)
    target_values = get_sawtooth_waves(sample_spacing,period,norm=True)
    rc_pipeline.define_target(target_values)

    # Define model parameters
    model_params = {
    "alpha": 1e-3 ,
    "fit_intercept": True ,
    "copy_X": True ,
    "max_iter": None ,
    "tol": 0.0001 ,
    "solver": "auto",
    "positive": False ,
    "random_state": None
    }

    # Define the training model
    model = define_Ridge(model_params)

    # Define the RC parameters
    rc_params = {
    "model": model ,
    "test_size": 0.3 ,
    "error_type": "MSE",
    "tau":0} # transformation task

    rc_pipeline.run(rc_params)

    results = rc_pipeline.get_rc_results()
    
    # Results
    train_ys = results["train"]["y_train"]
    test_ys = results["test"]["y_test"]
    train_preds = results["train"]["train_pred"]
    test_preds = results["test"]["test_pred"]
    train_MSE = results["error"]["train_error"]
    test_MSE = results["error"]["test_error"]
    
    train_ts = np.arange(train_ys.shape[0])
    test_ts = np.arange(test_ys.shape[0])

    # Errors
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
        
