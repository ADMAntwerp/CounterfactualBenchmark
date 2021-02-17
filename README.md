# Universal Counterfactual Benchmark Framework

## Run multiple shell
Example to run multiple without terminal message outputs
### Arguments
* A - Name of the benchmark to be run
* B - Dataset to be run (0 to 21)
* C - Class (0 or 1)
* D - Row number (depends of the dataset)
```shell script
cd ./benchmark
sh run_shell_multiple.sh A B C D &> /dev/null
```

Example:
```shell script
cd ./benchmark
sh run_shell_multiple.sh benchmark_MLEXPLAIN.py 0 0 0 &> /dev/null
```