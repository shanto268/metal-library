import numpy as np
import pandas as pd

from metal_library import logging
from metal_library.core.reader import Reader
from metal_library.core.sweeper_helperfunctions import create_dict_list

class Selector:

    __supported_metrics__ = ['Euclidean', 'Manhattan', 'Chebyshev', 'Weighted Euclidean' , 'Custom']
    __supported_estimation_methods__ = ['Interpolation']

    def __init__(self, reader):

        # Will be overwritten by `self.parseReader`
        self.component_type = None
        self.geometry = None
        self.characteristic = None
        self.custom_metric_func = None
        self.metric_weights = None
        
        if isinstance(reader, Reader):
            self.reader = reader
            self._parse_reader(reader) # Assigns: self.component_type, self.geometry, self.characteristic
        else:
            raise TypeError("`reader` must be `metal_library.Reader`")
    
    def _parse_reader(self, reader: Reader):
        """
        Extracts relevant library data from Reader

        Args: 
            reader (Reader)
        """
        if not (hasattr(reader.library, 'geometry') and hasattr(reader.library, 'characteristic')):
            raise AttributeError('`Reader` must have `Reader.library` created. Run `Reader.read_library` before initalizing `Selector`.')

        self.component_type = reader.library.component_type
        self.geometry = reader.library.geometry
        self.characteristic = reader.library.characteristic

    def _outside_bounds(self, df: pd.DataFrame, params: dict, display=True) -> bool:
        """
        Check to see if entered parameters are outside the bounds of a dataframe

        Args:
            df (pd.DataFrame): Dataframe to give warning.
            params (dict): Keys are column names of `df`. Values are values to check for bounds.
        
        Returns:
            bool: True if any value is outside of bounds. False if all values are inside bounds.
        """
        for param, value in params.items():
            if param in df.columns:
                if value < df[param].min() or value > df[param].max():
                    if display:
                        logging.info(f"NOTE TO USER: the value {value} for {param} is outside the bounds of our library.")
                        logging.info("If you find a geometry which corresponds to these values, please consider contributing it! 😁🙏")
                    return True
            else:
                raise ValueError(f"{param} is not a column in dataframe: {df}")
        
        return False


    def get_geometry_from_index(self, index: int) -> dict:
        """
        Get associated QComponent.options dictionary from index num.

        Args:
            index (int): Index of associated geometry.

        Returns:
            options (dict): Associated dictionary for QComponent.options
        """
        df = self.geometry.iloc[index]
        keys = list(df.keys())
        values = [list(df.values)]
        
        options = create_dict_list(keys=keys, values=values)[0]

        return options

    def find_closest(self,
                     target_params: dict,
                     num_top: int,
                     metric: str = 'Euclidean',
                     display: bool = True):
        """
        Main functionality. Select the closest presimulated geometry for a set of characteristics.
        
        Args:
            target_params (dict): A dictionary where the keys are the column names in `self.characteristic`,
                                  and the values are the target values to compare against.
            num_top (int): The number of rows with the smallest Euclidean distances to return.
            metric (str, optional): Metric to determine closeness. Defaults to "Euclidean". 
                                    Must choose from `self.__supported_metrics__`.
            display (boo, optional): Print out results? Defaults to True.

        Returns:
            indexes_smallest (pd.Index): Indexes of the 'num_top' rows with the smallest distances to the target parameters.
            best_characteristics (list[dict]): Associated characteristics. Ranked closest to furthest, same order as `best_geometries`
            best_geometries (list[dict]): Geometries in the style of QComponent.options. Ranked closest to furthest.

        """
        ### Checks
        # Check for supported metric
        if metric not in self.__supported_metrics__:
            raise ValueError(f'`metric` must be one of the following: {self.__supported_metrics__}')
        # Check for improper size of library
        if (num_top > len(self.characteristic)):
            raise ValueError('`num_top` cannot be bigger than size of read-in library.')
        # Log if parameters outside of library
        self._outside_bounds(df=self.characteristic, params=target_params, display=True)

        ### Setup
        # Choose from supported metrics, set it to var `find_index`
        if (metric == 'Euclidean'):
            find_index = self._find_index_Euclidean
        elif (metric == 'Manhattan'):
            find_index = self._find_index_Manhattan
        elif (metric == 'Chebyshev'):
            find_index = self._find_index_Chebyshev
        elif (metric == 'Weighted Euclidean'):
            find_index = self._find_index_Weighted_Euclidean
        elif (metric == 'Custom'):
            find_index = self._find_index_Custom_Metric

        ### Main Logic
        indexes_smallest = find_index(target_params=target_params, num_top=num_top)
        best_geometries = [self.get_geometry_from_index(index=index) for index in indexes_smallest]
        best_characteristics = [self.get_characteristic_from_index(index=index) for index in indexes_smallest]

        ### Print results in pretty format
        if display:
            # Formating
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
            END = '\033[0m'

            df_displayed = pd.DataFrame({
                "Ranking (Closest to Furthest)": range(1, len(best_characteristics) + 1),
                "Index": list(indexes_smallest),
                "Characteristic from Library": best_characteristics,
                "Geometry from Library": best_geometries
            })

            # Display / Printing
            print(f'{BOLD}{UNDERLINE}Here are the closest {num_top} geometries{END}')
            print(f'{BOLD}Target parameters:{END} {target_params}')
            print(f"{BOLD}Metric:{END} {metric}")

            from IPython.display import display, HTML
            display(HTML(df_displayed.to_html(index=False)))

        return indexes_smallest, best_characteristics, best_geometries

    def _find_index_Custom_Metric(self, target_params: dict, num_top: int):
        """
        Calculates the custom metric between each row in `self.characteristic` and a set of target parameters.
        It then returns the indexes of the 'num_top' rows with the smallest custom metric values.
        
        Parameters:
            target_params (dict): Dictionary of target Hamiltonian parameters.
            num_top (int): Number of top matches to return.
            custom_metric_func (callable): User-defined custom metric function. 
                                           The function should take two dictionaries as arguments and return a float.
            
        Returns:
            pd.Index: Indices of the 'num_top' closest matches based on the custom metric.
            
        Example Usage:
            To use a custom Manhattan distance metric, define the function as follows:
            
            def manhattan_distance(target, simulated):
                return sum(abs(target[key] - simulated.get(key, 0)) for key in target)
                
            Then, call `_find_index_custom_metric` with this function:
            
            closest_indices = selector._find_index_custom_metric(target_params, 2, manhattan_distance)
        """
        if self.custom_metric_func is None:
            raise ValueError('Must provide a custom metric function.')
        else:
            custom_metric_func = self.custom_metric_func
        
        distances = []
        for index, row in self.characteristic.iterrows():
            distance = custom_metric_func(target_params, row.to_dict())
            distances.append((index, distance))
        
        # Sort by distance and return the indices of the 'num_top' closest matches
        closest_indices = sorted(distances, key=lambda x: x[1])[:num_top]
        return pd.Index([index for index, _ in closest_indices])


    def _find_index_Weighted_Euclidean(self, target_params: dict, num_top: int):
        """
        Calculates the weighted Euclidean distance between each row in `self.characteristic` and a set of target parameters.
        It then returns the indexes of the 'num_top' rows with the smallest weighted Euclidean distances.
        
        Parameters:
            target_params (dict): Dictionary of target Hamiltonian parameters.
            num_top (int): Number of top matches to return.
            weights (dict, optional): Dictionary of weights for each parameter. Defaults to 1 for all if not provided.
            
        Returns:
            pd.Index: Indices of the 'num_top' closest matches based on weighted Euclidean distance.
        """
        if self.metric_weights is None:
            self.metric_weights = {key: 1 for key in target_params.keys()}

        
        distances = []
        for index, row in self.characteristic.iterrows():
            distance = 0
            for param, target_value in target_params.items():
                simulated_value = row.get(param, 0)
                weight = self.metric_weights.get(param, 1)
                distance += weight * ((target_value - simulated_value) ** 2) / target_value
                
            distances.append((index, distance))
        
        # Sort by distance and return the indices of the 'num_top' closest matches
        closest_indices = sorted(distances, key=lambda x: x[1])[:num_top]
        return pd.Index([index for index, _ in closest_indices])
    
    def get_characteristic_from_index(self, index: int) -> dict:
        """
        Get associated characteristics from index num.

        Args:
            index (int): Index of associated characteristic.

        Returns:
            options (dict): Associated dictionary for QComponent.options
        
        """
        df = self.characteristic.iloc[index]
        keys = list(df.keys())
        values = [list(df.values)]
        
        options = create_dict_list(keys=keys, values=values)[0]

        return options

    def _find_index_Euclidean(self, target_params: dict, num_top: int):
        """
        Calculates the Euclidean distance between each row in `self.characteristic` and a set of target parameters.
        It then returns the indexes of the 'num_top' rows with the smallest Euclidean distances.
        The Euclidean distance here is calculated as: sqrt(sum_i (x_i - x_{target})^2 / x_{target}),
        where x_i are the values in the DataFrame and x_{target} are the target parameters.

        Args:
            target_params (dict): A dictionary where the keys are the column names in `self.characteristic`,
                                  and the values are the target values to compare against.
            num_top (int): The number of rows with the smallest Euclidean distances to return.

        Returns:
            indexes_smallest (pd.Index): Indexes of the 'num_top' rows with the smallest Euclidean distances to the target parameters.
        """
        # Start with an array of zeros with the same length as the DataFrame
        distances = np.zeros(self.characteristic.shape[0])
        
        # Euclidean Metric
        for column, target_value in target_params.items():
            distances += ((self.characteristic[column] - target_value)**2 / target_value)
        distances = np.sqrt(distances)

        # Return the indexes of the rows with the smallest distances
        distances.sort_values(inplace=True)
        indexes_smallest =  distances.nsmallest(num_top).index

        return indexes_smallest

    def _find_index_Manhattan(self, target_params: dict, num_top: int):
        """
        Calculates the Manhattan distance between each row self.characteristic and a set of target parameters.
        It then returns the indexes of the 'num_top' rows with the smallest Manhattan distances.
        The Manhattan distance is calculated as: sum_i |x_i - x_{target}|,
        where x_i are the values in the DataFrame and x_{target} are the target parameters.

        Args:
            target_params (dict): A dictionary where the keys are the column names in self.characteristic,
                                  and the values are the target values to compare against.
                                
            num_top (int): The number of rows with the smallest Manhattan distances to return.

        Returns:
            indexes_smallest (pd.Index): Indexes of the 'num_top' rows with the smallest Manhattan distances to the target parameters.
        """
        # Start with an array of zeros with the same length as the DataFrame
        distances = np.zeros(self.characteristic.shape[0])
        
        # Manhattan Metric
        for column, target_value in target_params.items():
            distances += np.abs(self.characteristic[column] - target_value)
        
        # Return the indexes of the rows with the smallest distances
        distances.sort_values(inplace=True)
        indexes_smallest = distances.nsmallest(num_top).index

        return indexes_smallest
    
    def _find_index_Chebyshev(self, target_params: dict, num_top: int):
        """
        Calculates the Chebyshev distance between each row in self.characteristic and a set of target parameters.
        It then returns the indexes of the 'num_top' rows with the largest Chebyshev distances.
        The Chebyshev distance is calculated as: max_i |x_i - x_{target}|,
        where x_i are the values in the DataFrame and x_{target} are the target parameters.

        Args:
            target_params (dict): A dictionary where the keys are the column names self.characteristic,
                                  and the values are the target values to compare against.
                                
            num_top (int): The number of rows with the smallest Chebyshev distances to return.

        Returns:
            indexes_smallest (pd.Index): Indexes of the 'num_top' rows with the smallest Chebyshev distances to the target parameters.
        """
        # Initialize an array with a small value
        distances = np.full(self.characteristic.shape[0], -np.inf)
        
        # Chebyshev metric
        for column, target_value in target_params.items():
            distances = np.maximum(distances, np.abs(self.characteristic[column] - target_value))
        
        # Return the indexes of the rows with the smallest distances
        distances.sort_values(inplace=True)
        indexes_smallest =  distances.nsmallest(num_top).index

        return indexes_smallest
