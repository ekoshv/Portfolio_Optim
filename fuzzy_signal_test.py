import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzySignal:
    def __init__(self):
        # Define the membership functions for mx_val and mn_val
        self.mx_val = ctrl.Antecedent(np.arange(-10, 11, 1), 'mx_val')
        self.mn_val = ctrl.Antecedent(np.arange(-10, 11, 1), 'mn_val')
        
        self.state = ctrl.Consequent(np.arange(0, 9, 1), 'state')
        
        self.mx_val['low'] = fuzz.trapmf(self.mx_val.universe, [-10, -10, 0, 1.1])
        self.mx_val['medium'] = fuzz.trimf(self.mx_val.universe, [1.1, 2, 10])
        self.mx_val['high'] = fuzz.trapmf(self.mx_val.universe, [2, 10, 10, 10])

        self.mn_val['low'] = fuzz.trapmf(self.mn_val.universe, [-10, -10, -2, -1.1])
        self.mn_val['medium'] = fuzz.trimf(self.mn_val.universe, [-1.1, 0, 10])
        self.mn_val['high'] = fuzz.trapmf(self.mn_val.universe, [0, 10, 10, 10])

        self.state['0'] = fuzz.trimf(self.state.universe, [0, 0, 1])
        self.state['1'] = fuzz.trimf(self.state.universe, [1, 1, 2])
        # ... and so on for the other states

        # Define the rules
        rules = [
            ctrl.Rule(self.mx_val['high'] & self.mn_val['low'], self.state['0']),
            ctrl.Rule(self.mx_val['high'] & self.mn_val['medium'], self.state['1']),
            # ... and so on for the other rules
        ]

        # Create the control system and simulation
        self.signal_ctrl = ctrl.ControlSystem(rules)
        self.signal_simulation = ctrl.ControlSystemSimulation(self.signal_ctrl)

    def calculate_signal(self, fd):
        # Thresholding
        mx_val = min(fd.max(), 10)
        mn_val = max(fd.min(), -10)

        # Pass the input values to the simulation
        self.signal_simulation.input['mx_val'] = mx_val
        self.signal_simulation.input['mn_val'] = mn_val

        # Compute the fuzzy signal
        self.signal_simulation.compute()

        # Extract the output
        state = self.signal_simulation.output['state']
        sigs = [mx_val, mn_val]

        return state, sigs
