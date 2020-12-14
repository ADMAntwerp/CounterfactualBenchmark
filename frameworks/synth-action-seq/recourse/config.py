class Config(object):
    def __init__(self, learning_rate=0.005, adam_beta1=0.99, adam_beta2=0.999, adam_eps=1e-8,
                 binary_searches=9, init_c=1e5, c_upper_bound=1e10, iterations=200000,
                 starting_cost=1e10):
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.starting_cost = starting_cost
        self.binary_searches = binary_searches
        self.init_c = init_c
        self.c_upper_bound = c_upper_bound
        self.c_lower_bound = init_c
        self.iterations = iterations

    def __str__(self):
      a = ""
      a +=  '|||  learning_rate:  ' + str(self.learning_rate)
      a +=  '|||  adam_beta1:  ' + str(self.adam_beta1)
      a +=  '|||  adam_beta2:  ' + str(self.adam_beta2)
      a +=  '|||  adam_eps:  ' + str(self.adam_eps)
      a +=  '|||  iterations:  ' + str(self.iterations)
      return a



base_config = Config()

example_config_def = dict(learning_rate=0.01,
                          adam_beta1=0.9,
                          adam_beta2=0.999,
                          adam_eps=1e-5,
                          starting_cost=1e10,
                          binary_searches=10,
                          init_c=1e5,
                          c_upper_bound=1e10,
                          iterations=1000)
example_config = Config(**example_config_def)
