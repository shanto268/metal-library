import numpy as np
from qiskit_metal import designs, MetalGUI, Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander


def convert_numpy_to_python(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.number):
        return data.item()
    else:
        return data

class QubitCavity:
    def __init__(self, design, options):
        """
        Initialize a QubitCavity object with the provided design and options.
        
        Parameters:
        - design (DesignPlanar): An instantiated DesignPlanar object from Qiskit Metal.
        - options (Dict): Dictionary containing the options for the qubit, feedline, and cavity.
        """
        self.design = design
        self.options = options
        self._create_qubit()
        self._create_feedline()
        self._create_cavity()
        
    def _create_qubit(self):
        """
        Create a TransmonCross qubit based on the options provided.
        """
        qubit_options = self.options.get('qubit_options', {})
        if not qubit_options:
            print("Warning: qubit_options is empty. Using default settings.")
        
        self.qubit = TransmonCross(self.design, 'Q1', options=qubit_options)
        
    def _create_feedline(self):
        """
        Create a CoupledLineTee feedline based on the options provided.
        """
        coupler_options = self.options.get('cavity_options', {}).get('coupler_options', {})
        if not coupler_options:
            print("Warning: coupler_options is empty. Using default settings.")
        
        self.feedline = CoupledLineTee(self.design, 'feedline', coupler_options)
        


    def _create_cavity(self):
        """
        Create a RouteMeander cavity based on the options provided.
        """
        print("Entering _create_cavity method.")
        
        cpw_options = self.options.get('cavity_options', {}).get('cpw_options', {})
        if not cpw_options:
            print("Warning: cpw_options is empty. Using default settings.")
            return
    
        print(f"cpw_options before checks: {cpw_options}")
        
        # Check if pin_inputs exists and has necessary keys
        if 'pin_inputs' not in cpw_options or not all(key in cpw_options['pin_inputs'] for key in ['start_pin', 'end_pin']):
            print("Error: cpw_options['pin_inputs'] is missing essential attributes. Cavity will not be created.")
            return
    
        start_component = cpw_options['pin_inputs']['start_pin'].get('component', 'Unknown')
        end_component = cpw_options['pin_inputs']['end_pin'].get('component', 'Unknown')
    
        print(f"Start component: {start_component}, End component: {end_component}")
    
        # Check for 'nan' or None in pin_inputs and fill in default values based on qubit and feedline names
        if start_component in [None, 'nan', 'Unknown'] or end_component in [None, 'nan', 'Unknown']:
            print("Warning: cpw_options['pin_inputs'] contains ambiguous values. Filling with current qubit and feedline.")
            qubit_name = self.qubit.name
            feedline_name = self.feedline.name
            cpw_options['pin_inputs']['start_pin']['component'] = qubit_name
            cpw_options['pin_inputs']['start_pin']['pin'] = 'readout'  # Replace with the correct pin name
            cpw_options['pin_inputs']['end_pin']['component'] = feedline_name
            cpw_options['pin_inputs']['end_pin']['pin'] = 'second_end'  # Replace with the correct pin name
        
        # Convert NumPy types to native Python types
        cpw_options = convert_numpy_to_python(cpw_options)
    
        print(f"cpw_options after checks: {cpw_options}")
        
        self.cavity = RouteMeander(self.design, 'cavity', options=cpw_options)
    
        print("Exiting _create_cavity method.")




