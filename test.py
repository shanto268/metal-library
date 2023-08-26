best_geoms = selector.select_best_geometries(geometries, 3)

best_options = best_geoms[0]

QubitCavity(design, options=best_options)

class QubitCavity:

    def __init__(self, design, options=None):
        self.design = design
        self.options = options

    def select_cavity_type(self):
        if "custom" not in self.options.keys():
            return "route_meander"
        else:
            if self.options.custom == "inductive":
                return "inductive_coupler"

    def makeCavity(self):
        if self.select_cavity_type() == "route_meander":
            return self.make_route_meander()
        elif self.select_cavity_type() == "inductive_coupler":
            return self.make_inductive_coupler()

    def make_route_meander(self):
        return RouteMeander(self.design, options=self.options)

    def make_inductive_coupler(self):
        return InductiveCoupler(self.design, options=self.options)
