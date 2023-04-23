import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySignalCalculator:
    def __init__(self):
        # Create the fuzzy control system
        self.max_value = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'max_value')
        self.min_value = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'min_value')
        self.abs_value = ctrl.Antecedent(np.arange(0, 10, 0.1), 'abs_value')
        self.signal = ctrl.Consequent(np.arange(-4, 5, 1), 'signal')

        # Define the fuzzy sets
        self.max_value['low'] = fuzz.zmf(self.max_value.universe, -2, 1.5)
        self.max_value['medium'] = fuzz.trimf(self.max_value.universe, [1.03, 1.5, 2])
        self.max_value['high'] = fuzz.smf(self.max_value.universe, 1.5, 10)

        self.min_value['low'] = fuzz.zmf(self.min_value.universe, -10, -1.5)
        self.min_value['medium'] = fuzz.trimf(self.min_value.universe, [-2, -1.5, -1.03])
        self.min_value['high'] = fuzz.smf(self.min_value.universe, -1.5, 2)

        self.abs_value['low'] = fuzz.zmf(self.abs_value.universe, 0, 1.5)
        self.abs_value['medium'] = fuzz.trimf(self.abs_value.universe, [0, 1.5, 3])
        self.abs_value['high'] = fuzz.smf(self.abs_value.universe, 1.5, 3)

        self.signal.automf(names=['-2', '-1', '0', '1', '2'])

        # Define the fuzzy rules and create the control system
        self.create_rules()
        self.create_control_system()

    def create_rules(self):
        self.rules = [
            ctrl.Rule(self.max_value['low'] & self.min_value['low'], self.signal['0']),
            ctrl.Rule(self.max_value['medium'] & self.min_value['low'], self.signal['1']),
            ctrl.Rule(self.max_value['high'] & self.min_value['low'], self.signal['2']),
            ctrl.Rule(self.max_value['low'] & self.min_value['medium'], self.signal['-1']),
            ctrl.Rule(self.max_value['low'] & self.min_value['high'], self.signal['-2']),
            ctrl.Rule(self.max_value['medium'] & self.min_value['medium'], self.signal['0']),
            ctrl.Rule(self.max_value['high'] & self.min_value['medium'], self.signal['1']),
            ctrl.Rule(self.max_value['medium'] & self.min_value['high'], self.signal['-1']),
            ctrl.Rule(self.max_value['high'] & self.min_value['high'], self.signal['0'])
        ]

    def create_control_system(self):
        self.signal_ctrl = ctrl.ControlSystem(self.rules)
        self.signal_simulation = ctrl.ControlSystemSimulation(self.signal_ctrl)

    def calculate_signal(self, future_data_rescaled):
        max_value_input = future_data_rescaled.max()
        min_value_input = future_data_rescaled.min()

        self.signal_simulation.input['max_value'] = max_value_input
        self.signal_simulation.input['min_value'] = min_value_input
        self.signal_simulation.compute()

        return int(self.signal_simulation.output['signal'])
