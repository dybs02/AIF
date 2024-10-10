import numpy as np

MEAN_RANGE = (-10, 10)
VARIANCE_RANGE = (0.1, 1)


class DataGenerator(np.random.Generator):
    def __init__(self, seed=None):
        super().__init__(np.random.PCG64(seed))


    def rand_flaot(self, min=-1, max=1):
        return self.uniform(min, max)


    def generate_node(self, num_samples):
        return self.normal(
            loc=[self.rand_flaot(*MEAN_RANGE), self.rand_flaot(*MEAN_RANGE)],
            scale=[self.rand_flaot(*VARIANCE_RANGE), self.rand_flaot(*VARIANCE_RANGE)],
            size=[num_samples, 2]
            )


    def generate_data(self, num_nodes, num_samples):
        return {
            '0': [self.generate_node(num_samples) for _ in range(num_nodes)],
            '1': [self.generate_node(num_samples) for _ in range(num_nodes)]
        }