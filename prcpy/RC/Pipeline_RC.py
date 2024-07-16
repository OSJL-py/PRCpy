import numpy as np
import pandas as pd
from .Prepare_RC import Prepare
from .Perform_RC import Perform
from ..Maths.Maths_functions import get_sample_spacing, cov, estimator_capacity

from numpy.lib.stride_tricks import sliding_window_view

class Pipeline():
    """
    This class serves as the main interface for users to perform RC tasks. It handles
    data preparation, model setup, running the model, and retrieving results.
    """

    def __init__(self, data_dir_path: str, prefix: str, process_params: dict[str, any]):
        self.process_params = process_params
        self.prepared_data = None
        self.rc_ml = None
        
        self.rc_data = Prepare(data_dir_path,prefix)
        self.rc_data.create_experiment_df(Xs_idx=self.process_params["Xs"], Readouts_idx=self.process_params["Readouts"], delimiter=self.process_params["delimiter"])
        self.rc_data.process_data(**self.process_params)

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
        
    def setup_model(self, model: any) -> None:
        """
        Sets up the machine learning model to be used for RC.
        """
        self.model = model

    def run(self, rc_params: dict[str, any]) -> None:
        """
        Runs the model. Split the data, train the model, and evaluate the model.
        """
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
    
    ################ New functions #######################

    # def get_estimator_capacity(self) -> float:
    #     """ Measures the quality of a linear estimator from a multivariate series
    # 		to reconstruct a univariate series u

    # 	:param u : univariate series to reconstruct
    # 	:param X : multivariate series

    # 	:return : Measure depicting the quality of the estimator bound by the
    # 			  interval [0.0, 1.0].
    # 	"""

    #     u = self.rc_data.get_inputs()
    #     X = self.rc_data.get_readout_xs()

    #     n_train = int(u.shape[0] * 0.75)

    #     u_train = u[:n_train]
    #     X_train = X[:n_train]

    #     u_test  = u[n_train:]
    #     X_test  = X[n_train:]

    #     estimator = LinearRegression().fit(X_train, u_train) 

    #     u_pred = estimator.predict(X_test) # estimator reconstruction

    #     return cov(u_test, u_pred)**2 / (np.var(u) * np.var(u_pred))
    
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
        u = self.rc_data.rc_df
        X = self.rc_data.get_readout_xs()#self.rc_data.get_readout_xs()

        u_padded = np.pad(u, (k - 1), 'constant', constant_values=(0,0))

        u_history = sliding_window_view(u_padded, k, axis=1)[:len(u)] # axis??

        linearity = []

        for x in np.transpose(X):

             linearity.append(estimator_capacity(x, u_history))
             #linearity.append(estimator_capacity(u_history,x))

        return 1 - np.mean(linearity)

