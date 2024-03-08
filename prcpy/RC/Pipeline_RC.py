from RC.Prepare_RC import *
from RC.Perform_RC import *

class Pipeline():
    """
    This class serves as the main interface for users to perform RC tasks. It handles
    data preparation, model setup, running the model, and retrieving results.
    """

    def __init__(self, data_dir_path: str, process_params: dict[str, any]):
        self.process_params = process_params
        self.prepared_data = None
        self.rc_ml = None
        self.rc_data = Prepare(data_dir_path)
        self.rc_data.create_experiment_df(Xs_idx=self.process_params["Xs"], Readouts_idx=self.process_params["Readouts"])
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

