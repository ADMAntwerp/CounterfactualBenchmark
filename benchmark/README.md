## Full CF Benchmark
This folder has the benchmark algorithms for each original framework tested (ALIBIC, ALIBICNOGRAD, CADEX, DiCE, GROWINGSPHERES, LORE, MACE, MLEXPLAIN, SEDC, SYNAS).

The definitions for each algorithm is detailed in `benchmark_config.json`.

To make a full benchmark you must:

* Run the script `run_benchmark.sh` indicating the initial and final index that will be tested:
```shell script
bash run_benchmark.sh 0 99
```

As example above, it will run 100 (since final_index - initial_index + 1; 100 - 0 + 1) factual indexes to generate counterfactuals for all datasets (described in `../dataset_data/constants/var_types.py`) and all frameworks (described in `./benchmark_config.json`).

### WARNING
**It's important to highlight this benchmark run, for each dataset, all requested factual instances at the same time**. This means for the example above, 100 CF generation instances will be called **AT THE SAME TIME**, then, be careful and verify if your hardware can handle such run.

### If you are running from a personal computer, with low resources you may want run few instances

This could be, for example:

```shell script
bash run_benchmark.sh 0 2
```

That will just run 3 CF generation instances (for index 0, 1 and 2).