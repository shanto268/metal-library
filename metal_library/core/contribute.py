import csv
import numpy as np
import pandas as pd
import requests
from datetime import datetime

class CSVGenerator:
    """A class to generate a .csv file from qiskit metal design options, Hamiltonian parameters, and simulation info."""
    
    def __init__(self, design_options, H_params, sim_info, name="Data.csv"):
        """
        Initializes the CSVGenerator class.

        Parameters:
        design_options (dict): The design options from the qiskit metal design.
        H_params (dict): The Hamiltonian parameters for the system.
        sim_info (str): Miscellaneous simulation information.
        name (str):  The name of the .csv file to be generated. (Defaults to `Data.csv`)
        """
        self.name = name
        self.design_options = design_options
        self.H_params = H_params
        self.sim_info = sim_info

    def flatten_dict(self, d, parent_key='', sep='.'):
        """
        Flattens a nested dictionary structure.

        Parameters:
        d (dict): Dictionary to flatten.
        parent_key (str): The parent key for nested dictionary items.
        sep (str): The separator used in the flattened keys.

        Returns:
        dict: The flattened dictionary.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def generate_csv(self):
        """
        Generates the .csv file.
        """
        flat_design = self.flatten_dict(self.design_options)
    
        # Create a DataFrame from the flattened design dictionary
        df_design = pd.DataFrame([flat_design])
    
        # Create a DataFrame from the Hamiltonian parameters
        df_H_params = pd.DataFrame([self.H_params])
    
        # Create a DataFrame for the miscellaneous simulation info
        df_sim_info = pd.DataFrame({'misc': [self.sim_info]})
    
        # Concatenate the DataFrames horizontally and insert the "__SPLITTER__" column
        df_final = pd.concat([df_design, pd.DataFrame({'__SPLITTER__': ['split']}), df_H_params, df_sim_info], axis=1)
    
        # Fill NaN for empty cells
        df_final.fillna(value=np.nan, inplace=True)
    
        # Explicitly set data type for float columns if required
        for col in df_final.columns:
            if df_final[col].dtype == 'object':
                try:
                    df_final[col] = df_final[col].astype(float)
                except ValueError:
                    pass  # if conversion fails, keep as object
    
        # Write to CSV
        df_final.to_csv(self.name, index=False)


class SQuAADs:

    def __init__(self, design, H_params, sim_info, name="Data.csv"):
        """
        Parameters:
        design (qiskt metal design obj): The design from the qiskit metal design.
        H_params (dict): The Hamiltonian parameters for the system.
        sim_info (str): Miscellaneous simulation information.
        name (str):  The name of the .csv file to be generated. (Defaults to `Data.csv`)
        """
        self.name = name
        self.design = design
        self.H_params = H_params
        self.sim_info = sim_info

        # Create an instance of the QubitCavityCSVGenerator class
        csv_gen = CSVGenerator(design.options, H_params, sim_info, name)
        # Generate the QubitCavity.csv file
        csv_gen.generate_csv()


    def contribute(self, name, email, PI, institution):
            """
            Contribute the generated data.

            Parameters:
            name (str): Name of the contributor.
            email (str): Email of the contributor.
            PI (str): Principal Investigator.
            institution (str): Institution name.

            Returns:
            None
            """
            try:
                # Validate input
                if not all([name, email, PI, institution]):
                    raise ValueError("All parameters must be provided.")

                # Append datetime and contributor info to the file name
                dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
                new_file_name = f"{self.name}_{dt_str}_{name}.csv"

                # Renaming the file
                os.rename(self.name, new_file_name)

                # Send the data
                url = "YOUR_SERVER_URL_HERE"
                with open(new_file_name, 'rb') as f:
                    r = requests.post(url, files={'file': f})

                if r.status_code == 200:
                    print("Successfully contributed data.")
                else:
                    print(f"Failed to upload data. Server returned status code: {r.status_code}")

            except ValueError as ve:
                print(f"Value error: {ve}")
            except requests.exceptions.RequestException as re:
                print(f"Request error: {re}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
