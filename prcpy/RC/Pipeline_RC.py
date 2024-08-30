import numpy as np
import pandas as pd
from .Prepare_RC import Prepare
from .Perform_RC import Perform
from ..Maths.Maths_functions import get_sample_spacing, estimator_capacity, linear_memory_curve

from numpy.lib.stride_tricks import sliding_window_view

class Pipeline():
    """
    This class serves as the main interface for users to perform RC tasks. It handles
    data preparation, model setup, running the model, and retrieving results.
    """

    def __init__(self, data_dir_path: str, prefix: str, process_params: dict[str, any]):
        
        process_param_list = ["remove_bg", "smooth", "cut_xs", "normalize_local", "normalize_global", "sample"]

        if process_params["transpose"] == True:
            for item in process_param_list:
                if process_params[item] == True:
                    raise ValueError(f"transpose==True and {item}==True are not compatible.")
        
        self.process_params = process_params
        self.prepared_data = None
        self.rc_ml = None
        
        self.rc_data = Prepare(data_dir_path,prefix)
        self.rc_data_copy = Prepare(data_dir_path,prefix)
        
        self.rc_data.create_experiment_df(Xs_idx=self.process_params["Xs"], Readouts_idx=self.process_params["Readouts"], delimiter=self.process_params["delimiter"], transpose=self.process_params["transpose"])
        self.rc_data.process_data(**self.process_params)

        self.rc_data_copy.create_experiment_df(Xs_idx=self.process_params["Xs"], Readouts_idx=self.process_params["Readouts"], delimiter=self.process_params["delimiter"], transpose=self.process_params["transpose"])
        self.rc_data_copy.process_data(**self.process_params)

    def get_df_length(self) -> int:
        """
        Returns the length RC dataframe.
        """
        return self.rc_data.target_length

    def get_sample_spacing(self, period: int = 1) -> float:
        """
        Returns the sample spacing of the RC data.
        """
        return get_sample_spacing(self.rc_data.rc_df, period)

    def define_target(self, target: list[float]) -> None:
        """
        Defines the target values for the RC model.
        """
        self.rc_data.append_column("target", target)

    def define_input(self, input_data: np.array) -> None:
        """
        Defines the input values for the RC model (needed for the NL & MC metrics).
        """
        self.input_data = input_data
        
    def setup_model(self, model: any) -> None:
        """
        Sets up the machine learning model to be used for RC.
        """
        self.model = model

    def run(self, rc_params: dict[str, any]) -> None:
        """
        Runs the model. Split the data, train the model, and evaluate the model.
        """

        print("*****Running PRCpy*****")
        
        self.rc_ml = Perform(self.rc_data, rc_params)
        self.rc_ml.split_data()
        self.rc_ml.train_data()
        self.rc_ml.evaluate_data()

    def get_rc_results(self) -> dict[str, any]:
        """
        Returns the results of the model as dictionary.
        """
        if self.rc_ml is not None:
            return self.rc_ml.get_results()
        else:
            raise ValueError("Model has not been run. Please call the run method first.")
    
    def get_input_xs(self) -> np.array:
        """
        Returns the input values.
        """
        return np.array(self.rc_data.get_inputs())
    
    def get_readout_xs(self) -> np.array:
        """
        Returns the readout values.
        """
        return np.array(self.rc_data.get_readout_xs())
    
    def get_rc_df(self) -> pd.DataFrame:
        """
        Returns the full RC dataframe.
        """
        return self.rc_data.rc_df
    

    def get_non_linearity(self, k: int = 25) -> float:
        """ Measures the non-linearity of a system by approximating it as a LTI.
        The quality of the approximation is then measured to quantify the
        linearity of the system.

        :param u : reservoir input states
        :param X : reservoir output states

        :param kmax : maximum delay of LTI kernel

        :return : value determining the non-linearity of the system bound by the
                    interval [0.0, 1.0]
        """
        u = self.input_data

        if self.rc_data.transpose:
            X = np.array(self.rc_data_copy.rc_df).T
        else:
            X = np.array(self.rc_data_copy.rc_df)

        u_padded = np.pad(u, (k - 1), 'constant', constant_values=(0,0))
        u_history = sliding_window_view(u_padded, k)[:len(u)]

        linearity = []

        for x in np.transpose(X):
             linearity.append(estimator_capacity(x, u_history))

        return 1 - np.mean(linearity)

    def get_linear_memory_capacity(self, kmax: int = 25, remove_auto_correlation: bool = False) -> float:
        """ Linear memory capacity as defined by Herbert Jaeger

        :param u : reservoir input series
        :param X_res : reservoir output states
        :param X_auto : target values for auto correlation subtraction

        :return : total linear memory capacity
        """
        u = self.input_data
        
        if remove_auto_correlation:
            X_auto = self.input_data.reshape(-1,1)
        
        if self.rc_data.transpose:
            X_res = np.array(self.rc_data_copy.rc_df).T
        else:
            X_res = np.array(self.rc_data_copy.rc_df) 

        mc_res = linear_memory_curve(u, X_res, kmax)

        if remove_auto_correlation:
            mc_auto = linear_memory_curve(u, X_auto, kmax)
            mc = np.array(mc_res) - np.array(mc_auto)
        else:
            mc = mc_res

        return sum(mc), mc


