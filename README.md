# Universal Counterfactual Benchmark Framework

Fastest way to test your tabular counterfactuals, evaluating 22 different datasets/models. All models are Keras/TensorFlow NN.

## Installation

```bash
pip install cfbench
```

## Usage

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
benchmark_generator = BenchmarkCF(framework_name='my_framework').create_generator()

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
    evaluator(cf, verbose=True)
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

