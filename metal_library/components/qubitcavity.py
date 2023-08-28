from qiskit_metal import designs, MetalGUI, Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from metal_library import logging

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
            logging.info("Warning: qubit_options is empty. Using default settings.")
        
        self.qubit = TransmonCross(self.design, 'Q1', options=qubit_options)
        
    def _create_feedline(self):
        """
        Create a CoupledLineTee feedline based on the options provided.
        """
        coupler_options = self.options.get('cavity_options', {}).get('coupler_options', {})
        if not coupler_options:
            logging.info("Warning: coupler_options is empty. Using default settings.")
        
        self.feedline = CoupledLineTee(self.design, 'feedline', coupler_options)
        
    def _create_cavity(self):
        """
        Create a RouteMeander cavity based on the options provided.
        """
        cpw_options = self.options.get('cavity_options', {}).get('cpw_options', {})
        if not cpw_options:
            logging.info("Warning: cpw_options is empty. Using default settings.")
        
        self.cavity = RouteMeander(self.design, 'cavity', options=cpw_options)
