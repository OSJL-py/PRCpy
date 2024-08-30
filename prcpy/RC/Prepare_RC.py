"""
comment
"""
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter

from ..DataHandling.Path_handlers import get_full_paths
from ..DataHandling.File_handlers import load_csv
from ..Maths.Maths_functions import normalize_list

class Prepare():

    """
    Prepares the dataset for RC by processing and transforming the raw data.

    Attributes:
        root_path (str): The root directory path where the data files are located.
        full_path (list): A list of full paths to the data files.
        Xs_idx (str): The column name or index for the input variables in the dataset.
        Readouts_idx (str): The column name or index for the output variables (readouts) in the dataset.
        rc_df (pd.DataFrame): The processed DataFrame ready for RC.
        scan_cols (list): A list of column names in rc_df.
        readout_xs (list): A list of readout values extracted from the dataset.
    """
    
    def __init__(self, root_path: str, prefix: str):
        """
        Initializes the Prepare class with the root directory path of the data.

        Parameters:
            root_path (str): The root directory path.
        """

        self.root_path = root_path
        self.prefix = prefix
        self.full_path = get_full_paths(root_path,self.prefix)
        self.Xs_idx = ""
        self.Readouts_idx = ""
        self.rc_df = pd.DataFrame()
        self.scan_cols = []
        self.readout_xs = []
    
    def create_experiment_df(self, Xs_idx: str, Readouts_idx: str, delimiter: str = ",", transpose: bool = False) -> None:
        """
        Creates a DataFrame for the experiment by combining data from multiple files.

        Parameters:
            Xs_idx (str): The column name or index for the input variables in the dataset.
            Readouts_idx (str): The column name or index for the output variables (readouts) in the dataset.
        """

        dfs = []

        self.Xs_idx = Xs_idx
        self.Readouts_idx = Readouts_idx
        self.transpose = transpose

        for idx, fpath in enumerate(self.full_path, start=1):
            df = load_csv(fpath, delimiter=delimiter)[[Xs_idx, Readouts_idx]].rename(columns={Readouts_idx: f'{self.prefix}{idx}'})
            dfs.append(df)

        self.rc_df = pd.concat(dfs, axis=1).loc[:,~pd.concat(dfs, axis=1).columns.duplicated()]
        self.scan_cols = [col for col in self.rc_df.columns if "{}".format(self.prefix) in col]
        
        if self.transpose:
            self.target_length = self.rc_df.shape[0]
        else:
            self.target_length = self.rc_df.shape[1]

    def process_data(self, **kwargs: any) -> None:
        """
        Processes the data according to the provided keyword arguments. 
        Includes: background removal, smoothing, normalization, sampling, and slicing readouts by fault.

        Keyword Args:
            remove_bg (bool): If True, removes background signal from the data.
            bg_fname (str): The filename of the background data.
            smooth (bool): If True, applies a Savitzky-Golay filter to smooth the data.
            smooth_win (int): The window length for the Savitzky-Golay filter.
            smooth_rank (int): The polynomial order for the Savitzky-Golay filter.
            cut_xs (bool): If True, slices the data according to the provided x1 and x2 values.
            x1 (float): The lower bound for slicing the data.
            x2 (float): The upper bound for slicing the data.
            normalize (bool): If True, normalizes the data for each reservoir state.
            sample (bool): If True, samples the data according to the provided sample rate.
            sample_rate (int): The rate at which to sample the data.
        """

        if kwargs.get('remove_bg', False):
            bg_spec = load_csv(os.path.join(self.root_path, kwargs.get('bg_fname', "")))[self.Readouts_idx].values.squeeze()
            self.rc_df[self.scan_cols] = self.rc_df[self.scan_cols].sub(bg_spec, axis=0)

        if kwargs.get('smooth', False):
            self.rc_df[self.scan_cols] = self.rc_df[self.scan_cols].apply(savgol_filter, window_length=kwargs.get('smooth_win', 1), polyorder=kwargs.get('smooth_rank', 2), axis=0)

        if kwargs.get('cut_xs', False):
            x1, x2 = kwargs.get('x1', 0), kwargs.get('x2', 0)
            self.rc_df = self.rc_df[(self.rc_df[self.Xs_idx] >= x1) & (self.rc_df[self.Xs_idx] <= x2)]

        if kwargs.get('normalize_local', False):
            self.rc_df[self.scan_cols] = self.rc_df[self.scan_cols].apply(normalize_list)

        if kwargs.get('normalize_global', False):
            self.rc_df[self.scan_cols] = self.rc_df[self.scan_cols].apply(self.normalize_list_global)

        if kwargs.get('sample', False):
            self.rc_df = self.rc_df[::kwargs.get('sample_rate', 1)]

        if kwargs.get('transpose', False):
            self.transpose = True

        self.readout_xs = self.rc_df[self.Xs_idx].values
        self.rc_df = self.rc_df.drop(columns=[self.Xs_idx])

        self.transpose_df()
        #self.append_column("Inputs", get_raw_input_vals(self.root_path,self.prefix))
        self.define_rc_readout()
    
    def normalize_list_global(self, x, y1=0, y2=1):
        norm_list = (x - self.rc_df[self.scan_cols].values.flatten().min()) / (
                self.rc_df[self.scan_cols].values.flatten().max() - self.rc_df[
                self.scan_cols].values.flatten().min()) * (y2 - y1) + y1
        return norm_list

    def transpose_df(self) -> None:
        """
        Transposes the DataFrame.
        """
        self.rc_df = self.rc_df.transpose()

    def append_column(self, col_name: str, vals: list[any]) -> None:
        """
        Appends a column to the DataFrame.

        Parameters:
            col_name (str): The name of the column to append.
            vals (list[any]): The values to append to the column.
        """
        
        if self.transpose:
            self.rc_df = self.rc_df.T
            self.rc_df[col_name] = vals[:len(self.rc_df)]
            self.rc_df = self.rc_df.T
        else:
            self.rc_df[col_name] = vals[:len(self.rc_df)]

    def define_rc_readout(self) -> None:
        """
        Defines the readout columns for the RC DataFrame.
        """
        self.rc_readout = [f"r{col}" for col in self.rc_df.columns if type(col)==int]
        self.rc_df.columns = [self.rc_readout[i] if type(col) == int else col for i, col in enumerate(self.rc_df.columns)]

    def get_readout_xs(self) -> list[float]:
        """
        Returns the x values for the readouts.
        """
        return self.readout_xs
    
    def get_inputs(self) -> pd.Series:
        """
        Returns the input values from the experiment.
        """
        return self.rc_df["Inputs"]

