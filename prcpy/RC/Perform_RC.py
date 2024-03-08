import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Perform():
    """
    Performs RC including data splitting, training, and evaluation.

    Attributes:
        prepared_data (Prepare): An instance of the Prepare class containing the processed data.
        params (dict): A dictionary of parameters for model performance.
        model (Any): The machine learning model to be used.
        rc_readout (pd.Series): The readout values from the prepared data.
        targets (pd.Series): The target values for the model.
    """

    def __init__(self, prepared_data, params: dict[str, any]):
        """
        Parameters:
            prepared_data (Prepare): An instance of the Prepare class containing the processed data.
            params (dict[str, any]): A dictionary of parameters for model evaluation.
        """
        self.prepared_data = prepared_data
        self.params = params
        rc_df = prepared_data.rc_df
        self.model = self.params["model"]
        self.rc_readout = rc_df[self.prepared_data.rc_readout]
        self.targets = rc_df['target']

    def split_data(self) -> None:
        """
        Splits the data into training and testing sets based on the provided parameters.
        """
        def split_forecast():
            num_train = int(len(self.targets)*(1-self.params["test_size"]))
            tau = self.params["tau"]
            self.x_train, self.y_train = self.rc_readout[:num_train], self.targets[tau:num_train + tau]
            self.x_test, self.y_test = self.rc_readout[num_train:-tau], self.targets[num_train + tau:]

        def split_transformation():
            test_size = self.params["test_size"]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.rc_readout, self.targets, test_size=test_size, shuffle=False)

        if self.params["tau"] > 0:
            split_forecast()    
        else:
            split_transformation()

    def train_data(self) -> None:
        """
        Trains the model using the training data.
        """
        self.model_fit = self.model.fit(self.x_train, self.y_train)

    def predict_data(self, data) -> pd.Series:
        """
        Predicts the target values using the trained model.

        Parameters:
            data (pd.Series): The data to predict.

        Returns:
            pd.Series: The predicted target values.
        """
        y_pred = self.model_fit.predict(data)
        return y_pred
    
    def evaluate_data(self) -> None:
        """
        Evaluates the model using the training and testing data.
        """
        self.train_pred = self.predict_data(self.x_train)
        self.test_pred = self.predict_data(self.x_test)

    ## Getters
    ########################
    def get_results(self) -> dict[str, any]:
        """
        Returns the results of the model and performance in dictionary.
        """
        self.get_performance(self.params["error_type"])

        results_df = {
            "train": {
                "x_train": self.x_train,
                "y_train": self.y_train,
                "train_pred": self.train_pred
            },
            "test": {
                "x_test": self.x_test,
                "y_test": self.y_test,
                "test_pred": self.test_pred
            },
            "error": {
                "train_error": self.train_error,
                "test_error": self.test_error
            }
        }

        return results_df

    def get_performance(self, error_type: str) -> None:
        """
        Evaluates the model performance based on the provided error type.
        """
        self.train_error = 0
        self.test_error = 0

        if error_type == "MSE":
            self.train_error = mean_squared_error(self.y_train, self.train_pred)
            self.test_error = mean_squared_error(self.y_test, self.test_pred)

        elif error_type == "MAE":
            self.train_error = mean_absolute_error(self.y_train, self.train_pred)
            self.test_error = mean_absolute_error(self.y_test, self.test_pred)

    def get_fitted_model(self) -> any:
        """
        Returns the fitted model.
        """
        return self.model_fit

    def get_weights(self) -> pd.Series:
        """
        Returns the weights of the model.
        """
        return self.model_fit.coef_

    def get_intercept(self) -> float:
        """
        Returns the intercept of the model.
        """
        return self.model_fit.intercept_

    