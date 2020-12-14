Repository for "Synthesizing Optimal Action Sequences for Modifying Model Decisions" (Published in AAAI-2020)

[Link to paper](https://arxiv.org/abs/1910.00057)

### Setup
```shell
$ pip install -r requirements.txt
```

### Running the program
```shell
$ python run.py --target-model german \
                --ckpt model.h5 \
                --target-data data.npy \
                --mode vanilla \
                --l 4 \
                --actions WaitYears Naturalize ChangeCreditAmount ChangeLoanPeriod AdjustLoanPeriod GetGuarantor GetUnskilledJob \
```

### Description of repository contents
```
. <package root>
+-- actions     # contains definitions for the Action and Feature classes, and precondition helpers
+-- common      # contains logger / path definitions
+-- heuristics  # contains the code to implement the score functions
+-- models      # contains model specific data, including model/data classes, custom action definitions and data
    +-- <model>
        +-- actions.py      # list of model-specific action classes. Each action takes the entire features.json metadata
                              each action requires definition of following methods
                              - apply(): action transformation method. Can be used for both tensors and numpy arrays
                              - get_cost(): cost calculations. Can be used for both tensors and numpy arrays
                              - precondition():
                                - during optimization, conditions are converted to continuous functions and added to the cost
                                - during condition check, the conditions output boolean outputs
        +-- dataset.py      # wrapper class for npy dataset, as well as feature meta-info extraction routine
        +-- features.json   # json definition for each feature in dataset
        +-- model.h5        # model checkpoint file in h5 format
        +-- model.py        # wrapper class to load (and train/save) each model
        +-- data.npy        # numpy array containing 100 random negatively classified points from the test set
    +-- adult
    +-- fanniemae
+-- recourse    # main search module
    +-- config.py           # contains configuration
    +-- result.py           # contains classes to store instance and sequence level results
    +-- search.py           # main search algorithm - contains two classes for search:
                            - SequenceSearch for outer-loop that selects the next sequence to optimize
                            - ParamSearch for inner-loop that performs the minimization
    +-- utils.py            # miscellaneous utilities
```

Sample results from running the following command can be found in the results folder:
```shell
$ python .\run.py --target-model german --ckpt model.h5 --mode vanilla --exp-name sample_length_1 --l 1
```




