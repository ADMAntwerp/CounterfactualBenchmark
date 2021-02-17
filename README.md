# Universal Counterfactual Benchmark Framework

## Run multiple shell
Example to run multiple without terminal message outputs
### Arguments
* A - Starting Row 
* B - Ending Row
* C - Name of the benchmark to be run
* D - Dataset to be run (0 to 21)
* E - Class (0 or 1)
```shell script
cd ./benchmark
sh run_shell_multiple.sh A B C D E &> /dev/null
```

Example:
```shell script
cd ./benchmark
sh run_shell_multiple.sh 0 10 benchmark_MLEXPLAIN.py 0 0 &> /dev/null
```

### WARNING

**ONLY USE THE SCRIPTS `run_full_batch_DS0.sh` AND `run_full_batch_DS1.sh` IN A POWERFUL COMPUTER, THESE SCRIPTS RUN ALL DATASETS IN ONE TIME FOR ONE CLASS**

Example:
```shell script
sh run_full_batch_DS0.sh benchmark_MLEXPLAIN.py &> /dev/null
```