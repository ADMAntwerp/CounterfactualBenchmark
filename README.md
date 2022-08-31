# Universal Counterfactual Benchmark Framework

Fastest way to test your tabular counterfactuals, evaluating 22 different datasets/models. All models are Keras/TensorFlow NN.

## Ranking
You can see the ranking of the best counterfactual explanation generation algorithms in this repository: https://github.com/rmazzine/Ranking-Tabular-CF

## Installation

```bash
pip install cfbench
```

## Usage
This code will just run the counterfacutal generator ``my_cf_generator`` on all factual instances and models.
Not creating any data, analysis or submitting to the benchmark. 
If you want to do that, see the examples in Further Examples.

```python
import numpy as np
from cfbench.cfbench import BenchmarkCF

# A simple CF generator, when the factual class is 1
# return full 0 array, otherwise return full 1 array
def my_cf_generator(factual_array, model):
    if model.predict(np.array([factual_array]))[0][0] > 0.5:
        return [0]*len(factual_array)
    else:
        return [1]*len(factual_array)

# Create Benchmark Generator
benchmark_generator = BenchmarkCF().create_generator()

# The Benchmark loop
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']
    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Create CF
    cf = my_cf_generator(factual_array, model)

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']
    # Evaluate CF
    evaluator(cf, verbose=True, algorithm_name="my_cf_generator")

```

## Further information
We understand that different counterfactual generators need different data, so our generator provide multiple data described in the following table:
<details>
  <summary>Click here for detailed info</summary>

The ``BenchmarkCF().create_generator()`` method returns a generator that provides the following data:

| key               | Type                              | Description                                                                                           |
|-------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| **factual_oh**    | list                              | Factual, one hot encoded (if categorical features), data                                              |
| **model**         | tf.Keras.Model                    | Model to be explained                                                                                 |
| **factual**       | list                              | Factual data (WITHOUT one hot encoding)                                                               |
| **num_feats**     | list                              | Indexes of the numerical continuous features                                                          |
| **cat_feats**     | list                              | Indexes of the categorical features                                                                   |
| **cf_evaluator**  | BenchmarkGenerator.cf_evaluator   | Evaluates if the CF is indeed a CF. Returns [True, cf_array] if a CF and [False, nan_array] otherwise |
| **oh_converter**  | cfbench.cfg.OHConverter.Converter | Converts to one hot ``.convert_to_oh`` or from one hot ``.convert``                                   |
| **df_train**      | pandas.DataFrame                  | Dataframe of model's training data (WITHOUT one hot encoding)                                         |
| **df_oh_train**   | pandas.DataFrame                  | Dataframe of model's training data (WITH one hot encoding)                                            |
| **df_test**       | pandas.DataFrame                  | Dataframe of model's test data (WITHOUT one hot encoding)                                             |
| **df_oh_test**    | pandas.DataFrame                  | Dataframe of model's test data (WITH one hot encoding)                                                |
| **df_factual**    | pandas.DataFrame                  | Dataframe of factual data (WITHOUT one hot encoding)                                                  |
| **tf_session**    | tf.Session                        | TensorFlow session                                                                                    |
| **factual_idx**   | int                               | Index of the factual data in the factual dataset                                                      |
| **factual_class** | int                               | Model's prediction (0 or 1) of the factual data                                                       |
| **dsname**        | str                               | Name of the dataset                                                                                   |


    
</details>

### I want to get general metrics of my counterfactual generator
If you want to get general metrics (coverage, sparsity, l2, mean absolute deviation, Mahalanobis distance, and
generation time), you can use the sample code below

<details>
  <summary>Click here to see the code</summary>

To generate a global analysis, you must create experiment data with the evaluator (``benchmark_data['cf_evaluator']``)
and assigning the ``cf_generation_time`` (the time it took to generate the CF) and ``save_results`` as True, to 
create the data to be analyzed (that will be in the folder ``./cfbench_results``.)

Then, the ``analyze_results`` method makes the global analysis and returns a dataframe with all
results processed in the folder ``./cfbench_results_processed/``. And it also prints the global
metrics in the console.

```python
import time
import numpy as np
from cfbench.cfbench import BenchmarkCF, analyze_results

# A simple CF generator, when the factual class is 1
# return full 0 array, otherwise return full 1 array
def my_cf_generator(factual_array, model):
    if model.predict(np.array([factual_array]))[0][0] > 0.5:
        return [0]*len(factual_array)
    else:
        return [1]*len(factual_array)

# Create Benchmark Generator
benchmark_generator = BenchmarkCF().create_generator()

# The Benchmark loop
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']
    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Create CF measuring how long it takes to generate the CF
    start_time = time.time()
    cf = my_cf_generator(factual_array, model)
    cf_generation_time = time.time() - start_time

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']
    # Evaluate CF
    evaluator(
        cf_out=cf,
        algorithm_name='my_cf_generator',
        cf_generation_time=cf_generation_time,
        save_results=True)

analyze_results('my_cf_generator')
```

</details>

### I want to rank my algorithm
If you want to compare your algorithm with others, you can use the code below.
<details>
  <summary>Click here to see the code</summary>

To correctly send the results, you must create experiment data with the evaluator (``benchmark_data['cf_evaluator']``)
and assigning the ``cf_generation_time`` (the time it took to generate the CF) and ``save_results`` as True, to 
create the data to be sent (that will be in the folder ``./cfbench_results``.)

After the experiment loop, you must call the ``send_results`` method of the evaluator,
to send the results to the server.

This function will also create in the folder ``./cfbench_results_processed/`` a file with the
processed results of your algorithm.

```python
import time
import numpy as np
from cfbench.cfbench import BenchmarkCF, send_results

# A simple CF generator, when the factual class is 1
# return full 0 array, otherwise return full 1 array
def my_cf_generator(factual_array, model):
    if model.predict(np.array([factual_array]))[0][0] > 0.5:
        return [0]*len(factual_array)
    else:
        return [1]*len(factual_array)

# Create Benchmark Generator
benchmark_generator = BenchmarkCF().create_generator()

# The Benchmark loop
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']
    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Create CF measuring how long it takes to generate the CF
    start_time = time.time()
    cf = my_cf_generator(factual_array, model)
    cf_generation_time = time.time() - start_time

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']
    # Evaluate CF
    evaluator(
        cf_out=cf,
        algorithm_name='my_cf_generator',
        cf_generation_time=cf_generation_time,
        save_results=True)

send_results('my_cf_generator')
```

After making the experiments and creating the analysis, you must fork [this repository](https://github.com/rmazzine/Ranking-Tabular-CF).

Then, you must provide the SSH path to your forked repo and, then, finally make a pull request to the main repository.

All these details are included in the algorithm, in a step-by-step process.
    
</details>



## TensorFlow Version compatibility
This framework is supposed to be compatible with TensorFlow 1 and 2, however, problems can arise. Therefore, 
if you encounter any problem, please open an issue.

## Reference
If you used this package on your experiments, here's the reference paper:
```bibtex
@Article{app11167274,
AUTHOR = {de Oliveira, Raphael Mazzine Barbosa and Martens, David},
TITLE = {A Framework and Benchmarking Study for Counterfactual Generating Methods on Tabular Data},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {16},
ARTICLE-NUMBER = {7274},
URL = {https://www.mdpi.com/2076-3417/11/16/7274},
ISSN = {2076-3417},
DOI = {10.3390/app11167274}
}
```

