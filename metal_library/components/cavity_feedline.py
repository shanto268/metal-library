import numpy as np
from qiskit_metal import Dict
from qiskit_metal.qlibrary.core import QRoute, QRoutePoint
from qiskit_metal.qlibrary.core import QComponent

class CavityFeedline(QComponent):
    
    default_options = dict(
        coupling_type = 'capacitive',
        coupler_options = dict(
                        #    prime_width='10um',
                        #    prime_gap='6um',
                        #    second_width='10um',
                        #    second_gap='6um',
                        #    coupling_space='3um',
                        #    coupling_length='100um',
                        #    down_length='100um',
                        #    fillet='25um',
                        #    mirror=False,
                        #    open_termination=True,
                        ),
        cpw_options = dict(
                            total_length='6000um',
                            # pin_inputs = dict(
                            #     start_pin = dict(
                            #         component = None,
                            #         pin = None
                            #     ),
                            #     end_pin = dict(
                            #         component = None,
                            #         pin = None
                            #     ),
                            # ),
                            left_options = dict(
                                meander=dict(spacing='100um', asymmetry='0um'),
                                # snap='true',
                                # prevent_short_edges='true',
                                fillet = '49.9um',
                            ),
                            right_options = dict(
                                meander=dict(spacing='100um', asymmetry='0um'),
                                # snap='true',
                                # prevent_short_edges='true',
                                fillet = '49.9um',
                            )
                        ),
        mirror=False
    )
    
    component_metadata = dict(short_name='cavity')
    """Component metadata"""

    # default_options = dict(meander=dict(spacing='200um', asymmetry='0um'),
    #                        snap='true',
    #                        prevent_short_edges='true')
    """Default options"""

    def copier(self, d, u):
        for k, v in u.items():
            if not isinstance(v, str) and not isinstance(v, float):
                d[k] = self.copier(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def make(self):
        p = self.p
        self.make_coupler()
        self.make_pins()
        self.make_cpws()


    def make_coupler(self):
        p = self.p

        temp_opts = dict()
        for k in p.coupler_options:
            temp_opts.update({k:p.coupler_options[k]})

        if(p.coupling_type == "capacitive"):
            from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
            self.coupler = CoupledLineTee(self.design, "{}_cap_coupler".format(self.name), options=temp_opts)
        elif(p.coupling_type == 'inductive'):
            from inductive_coupler import InductiveCoupler
            self.coupler = InductiveCoupler(self.design, "{}_ind_coupler".format(self.name), options=temp_opts)
        # self.add_qgeometry('path', self.coupler.qgeometry_dict('path'))
        # self.add_qgeometry('poly', self.coupler.qgeometry_dict('poly'))

    def make_cpws(self):
        from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

        p = self.p
        
        left_opts = dict()
        left_opts.update({'total_length': (p.cpw_options.total_length if p.coupling_type == 'capacitive' else p.cpw_options.total_length/2) })
        left_opts.update({'pin_inputs':dict(
                                            start_pin = dict(
                                                component = '',
                                                pin = ''
                                            ),
                                            end_pin = dict(
                                                component = '',
                                                pin = ''
                                            )
                                            )})
        left_opts['pin_inputs']['start_pin'].update({'component':p.cpw_options.pin_inputs.start_pin.component})
        left_opts['pin_inputs']['start_pin'].update({'pin':p.cpw_options.pin_inputs.start_pin.pin})

        left_opts['pin_inputs']['end_pin'].update({'component':self.coupler.name})
        left_opts['pin_inputs']['end_pin'].update({'pin':'second_end'})

        self.copier(left_opts, p.cpw_options.left_options)

        LeftMeander = RouteMeander(self.design, "{}_left_cpw".format(self.name), options = left_opts)
        # self.add_qgeometry('path', LeftMeander.qgeometry_dict('path'))
        # self.add_qgeometry('poly', LeftMeander.qgeometry_dict('poly'))

        if(p.coupling_type == 'inductive'):
            right_opts = dict()
            right_opts.update({'total_length':p.cpw_options.total_length/2})
            right_opts.update({'pin_inputs':dict(
                                                start_pin = dict(
                                                    component = '',
                                                    pin = ''
                                                ),
                                                end_pin = dict(
                                                    component = '',
                                                    pin = ''
                                                )
                                                )})
            right_opts['pin_inputs']['end_pin'].update({'component':p.cpw_options.pin_inputs.end_pin.component})
            right_opts['pin_inputs']['end_pin'].update({'pin':p.cpw_options.pin_inputs.end_pin.pin})

            right_opts['pin_inputs']['start_pin'].update({'component':self.coupler.name})
            right_opts['pin_inputs']['start_pin'].update({'pin':'second_start'})

            self.copier(right_opts, p.cpw_options.right_options)

            RightMeander = RouteMeander(self.design, "{}_right_cpw".format(self.name), options = right_opts)
            # self.add_qgeometry('path', RightMeander.qgeometry_dict('path'))
            # self.add_qgeometry('poly', RightMeander.qgeometry_dict('poly'))

        
    def make_pins(self):
        start_dict = self.coupler.get_pin('prime_start')
        end_dict = self.coupler.get_pin('prime_end')
        self.add_pin('prime_start', start_dict['points'], start_dict['width'])
        self.add_pin('prime_end', end_dict['points'], end_dict['width'])
