import numpy as np

class WeightedSampler:
    def __init__(self, values, dist_params):
        self.values = values
        self.dist_params = dist_params
        
        if self.dist_params.get('seed') is not None:
            np.random.seed(self.dist_params.get('seed'))

    def sample(self):
        dist_type = self.dist_params.get('type')
        if dist_type == 'uniform':
            return self.uniform_weighting()
        elif dist_type == 'normal':
            return self.normal_weighting()
        elif dist_type == 'magnitude':
            return self.magnitude_weighting()
        elif dist_type == 'binary':
            return self.binary_weighting()
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def set_max_min(self, min_val, max_val, values):
        m = (max_val - min_val)/(np.max(values) - np.min(values))
        b = min_val - m*(np.min(values))
        return [float(int(m*val+b)) for val in values]

    def uniform_weighting(self):
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        spread = self.dist_params.get('spread') # positive integer
        if len(self.values) == 1: 
            return float(max_val)
        elif len(self.values) == 0:
            return None
        else:
            values = list(np.random.randint(0, spread, size=len(self.values)))
            return self.set_max_min(min_val, max_val, values)

    def normal_weighting(self):
        # Return positive values from normal distribution
        # Similar to Normal, but sparser
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        sigma = self.dist_params.get('sigma') # positive float
        if len(self.values) == 1:
            return float(max_val)
        elif len(self.values) == 0:
            return None
        else:
            values = np.abs(np.random.normal(0, sigma, size=len(self.values)))
            return self.set_max_min(min_val, max_val, values)

    def magnitude_weighting(self):
        # Larger magnitudes weighted more
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        kT = self.dist_params.get('kT') # float
        if len(self.values) == 1:
            return float(max_val)
        elif len(self.values) == 0:
            return None
        else:
            starting_values = np.exp(np.array(np.abs(self.values))/kT)
            shift_values = np.random.normal(0, np.min(starting_values)*0.68, size=len(self.values))
            values = [starting_values[i] + shift_values[i] for i in range(len(starting_values))]
            return self.set_max_min(min_val, max_val, values)

    def binary_weighting(self):
        # One value or other
        min_val = self.dist_params.get('min')
        max_val = self.dist_params.get('max')
        split = self.dist_params.get('split') # positive float between 0 and 1
        choose_from = [min_val for i in range(int(100*split))] + [max_val for i in range(int(100*(1-split)))]
        if len(self.values) == 1:
            return float(max_val)
        elif len(self.values) == 0:
            return None
        else:
            assignments = list(np.random.randint(low=0, high=99, size=len(self.values)))
            return [float(choose_from[i]) for i in assignments]


