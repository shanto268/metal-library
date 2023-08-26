
# Qiskit Metal imports
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict
from metal_library.components import QubitCavity

design = designs.DesignPlanar()

gui = MetalGUI(design)


# Parsing the best geometries 
best_options        = best_geoms[0]
second_best_options = best_geoms[1]
# and so on...
options_qubit = Dict()

best_options = [options_qubit, options_coupler]

design = QubitCavity(design, options=best_options)


gui.rebuild()
gui.autoscale()
gui.screenshot()
