import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzySignal:
    def __init__(self):
        # Define the membership functions for mx_val and mn_val
        self.mx_val = ctrl.Antecedent(np.arange(-10, 11, 1), 'mx_val')
        self.mn_val = ctrl.Antecedent(np.arange(-10, 11, 1), 'mn_val')
        
        self.state = ctrl.Consequent(np.arange(0, 9, 1), 'state')
        
        self.mx_val['low'] = fuzz.zmf(self.mx_val.universe, -6, 1.5)
        self.mx_val['medium'] = fuzz.trimf(self.mx_val.universe, [1.1, 1.5, 2])
        self.mx_val['high'] = fuzz.smf(self.mx_val.universe, 1.5, 6)

        self.mn_val['low'] = fuzz.zmf(self.mn_val.universe, -6, -1.5)
        self.mn_val['medium'] = fuzz.trimf(self.mn_val.universe, [-2, -1.5, -1.1])
        self.mn_val['high'] = fuzz.smf(self.mn_val.universe, -1.5, 6)
        # Define the membership functions for state
        self.state['0'] = fuzz.trimf(self.state.universe, [0, 0, 1])
        self.state['1'] = fuzz.trimf(self.state.universe, [0, 1, 2])
        self.state['2'] = fuzz.trimf(self.state.universe, [1, 2, 3])
        self.state['3'] = fuzz.trimf(self.state.universe, [2, 3, 4])
        self.state['4'] = fuzz.trimf(self.state.universe, [3, 4, 5])
        self.state['5'] = fuzz.trimf(self.state.universe, [4, 5, 6])
        self.state['6'] = fuzz.trimf(self.state.universe, [5, 6, 7])
        self.state['7'] = fuzz.trimf(self.state.universe, [6, 7, 8])
        self.state['8'] = fuzz.trimf(self.state.universe, [7, 8, 9])

        # Define the rules
        rules = [
            ctrl.Rule(self.mx_val['high'] & self.mn_val['low'], self.state['0']),
            ctrl.Rule(self.mx_val['high'] & self.mn_val['medium'], self.state['1']),
            ctrl.Rule(self.mx_val['high'] & self.mn_val['high'], self.state['2']),
            ctrl.Rule(self.mx_val['medium'] & self.mn_val['low'], self.state['3']),
            ctrl.Rule(self.mx_val['medium'] & self.mn_val['medium'], self.state['4']),
            ctrl.Rule(self.mx_val['medium'] & self.mn_val['high'], self.state['5']),
            ctrl.Rule(self.mx_val['low'] & self.mn_val['low'], self.state['6']),
            ctrl.Rule(self.mx_val['low'] & self.mn_val['medium'], self.state['7']),
            ctrl.Rule(self.mx_val['low'] & self.mn_val['high'], self.state['8'])
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

# Create a sample dataset (fd) for demonstration
fd = np.array([-0.5, 1.8, 2.3, -2.3, 0.2])

# Instantiate the FuzzySignal class
fuzzy_signal = FuzzySignal()

# Call the calculate_signal method with the sample dataset
state, sigs = fuzzy_signal.calculate_signal(fd)

# Print the results
print(f"State: {state}")
print(f"mx_val: {sigs[0]}, mn_val: {sigs[1]}")