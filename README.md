# Universal Counterfactual Benchmark Framework

**A well-detailed tutorial on how to use this framework to test counterfactual generators 
can be found in:** https://mazzine.medium.com/testing-counterfactual-generation-algorithms-905f5c45fc1c

## Simple test tutorial
The example below shows how to run a simple test using all datasets (generating 10 counterfactuals per dataset, total 210).

### STEP 0 - Computer Requirements
To run this benchmark to test your counterfactual generation algorithm you will need:
* Ubuntu 18.04
* Anaconda 2020 version or later

### SETP 1 - Clone repository

Clone this repository using:
```shell script
git clone 
``` 

### SETP 2 - Modify `simple_test.py` SCRIPT
The script `simple_test.py` has the instructions on how you should add your CF generator. Modify it to include your algorithm.

*You may want to copy and paste your code in the same folder
*This script includes a dummy counterfactual generator (that only returns the factual instance), so you can run it first to understand more the framework and verify if the prerequisites are met.

There are 6 fields to be modified:
* Your framework name
* Dataset selection to be tested (categorical, numerical, mixed)
* Number of outputs from the neuronal network (1 or 2)
* Initial configuration of the counterfactual generator
* Generation of the counterfactual for the `factual` variable instance
* Post-processing of the counterfactual generator result, output of counterfactual candidate



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

## Instructions using on Google Cloud Computing Engine
For this experiment, the Google Cloud Computing Engine was used with the following specification:
### For datasets except InternetAdv
* Series: N1
* Machine Type: Custom
* Cores: 52
* CPU Platform: Intel Skylake or later
* Memory: 195 GB
* OS: Ubuntu 18.04
* Disk: SSD 400 GB

### For InternetAdv dataset
* Series: N2D
* Machine Type: Custom
* Cores: 48
* CPU Platform: AMD Rome or later
* Memory: 384 GB
* OS: Ubuntu 18.04
* Disk: SSD 400 GB

### Benchmark steps on GCCE
##### **WARNING - THE FOLLOWING STEPS WILL RUN A SCRIPT THAT CONSUMES LOTS OF RESOURCES, SEE THE COMPUTING REQUIREMENTS BEFORE USING**

Use as root
```shell script
sudo su
```

Go to temporary folder
```shell script
cd /tmp
```

Download Anaconda 2020.11
```shell script
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

Install Anaconda
```shell script
bash Anaconda3-2021.11-Linux-x86_64.sh
```

Update source
```shell script
source ~/.bashrc
```

Clone this repo
```shell script
git clone ...
```

Enter repo benchmark folder
```shell script
cd CounterfactualBenchmark/benchmark
```

Update Ubuntu Package Manager
```shell script
apt-get update
```

Start run detached from current terminal (as it takes a long time to run)
```shell script
nohup bash run_benchmark_full_0_50.sh &> /dev/null
```

The next steps are made to guarantee the job will not stop even if the terminal session closes

Press Ctrl+Z to make the process in background

Return process run
```shell script
bg
```

Find process id (`PROCESS_ID`)
```shell script
jobs -l
```

Detach job from terminal session
```shell script
disown PROCESS_ID
```