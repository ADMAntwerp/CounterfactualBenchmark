#!/bin/bash

## Initial Setup with required Software
#echo "Initializing installations"
#if [ $(id -u) -eq 0 ]; then
#  # Sudo user
#  apt install build-essential -y &&
#  apt install wine64 -y &&
#else
#  # Non-sudo user
#  sudo apt install build-essential -y &&
#  sudo apt install wine64 -y &&
#fi
#
## Create All Conda Environments Needed
#echo "Creating conda environments"
#conda create --name ALIBIC python=3.7 -y &&
#conda activate ALIBIC &&
#pip install -r ../framework_requirements/alibic_requirements.txt &&
#
#conda create --name CADEX python=3.6 -y &&
#conda activate CADEX &&
#pip install -r ../framework_requirements/cadex_requirements.txt &&
#
#conda create --name DICE python=3.7 -y &&
#conda activate DICE &&
#pip install -r ../framework_requirements/dice_requirements.txt &&
#
#conda create --name GROWINGSPHERES python=3.6 -y &&
#conda activate GROWINGSPHERES &&
#pip install -r ../framework_requirements/growingspheres_requirements.txt &&
#
#conda create --name LORE python=3.7 -y &&
#conda activate LORE &&
#pip install -r ../framework_requirements/lore_requirements.txt &&
#
#conda create --name MACE python=3.6 -y &&
#conda activate MACE &&
#pysmt-install --z3 --confirm-agreement &&
#pip install -r ../framework_requirements/mace_requirements.txt &&
#
#conda create --name MLEXPLAIN python=3.7 -y &&
#conda activate MLEXPLAIN &&
#pip install -r ../framework_requirements/mlexplain_requirements.txt &&
#
#conda create --name SEDC python=3.7 -y &&
#conda activate SEDC &&
#pip install -r ../framework_requirements/sedc_requirements.txt &&
#
#conda create --name SYNAS python=3.7 -y &&
#conda activate SYNAS &&
#pip install -r ../framework_requirements/synas_requirements.txt &&

# Run all class 0 Datasets
# BalanceScale Class 0
# 1 - Initial row number
# 2 - Final row number
# 3 - Framework run algorithm
# 4 - Dataset 1
# 5 - Class
# 6 - Timeout
# 7 - Dataset 2
# 8 - Dataset 3
# 9 - Dataset 4
run_experiments_dataset () {
  touch ./log_bench/$3_$4.log;
  rm ./log_bench/$3_$4.log;
  touch ./log_bench/$3_$4.log;
  touch ./log_bench/$3_$7.log;
  rm ./log_bench/$3_$7.log;
  touch ./log_bench/$3_$7.log;
  touch ./log_bench/$3_$8.log;
  rm ./log_bench/$3_$8.log;
  touch ./log_bench/$3_$8.log;
  touch ./log_bench/$3_$9.log;
  rm ./log_bench/$3_$9.log;
  touch ./log_bench/$3_$9.log;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell.sh $3 $4 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell.sh $3 $7 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell.sh $3 $8 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell.sh $3 $9 $5 $DSIDX &
  done;
  init_exp_date=$(date +%s)
  partial_exp_date=$(date +%s)

  while [ $(( $partial_exp_date - $init_exp_date )) -lt $6 ]
  do
    total_lines_ds1=$(wc -l < ./log_bench/$3_$4.log)
    total_lines_ds2=$(wc -l < ./log_bench/$3_$7.log)
    total_lines_ds3=$(wc -l < ./log_bench/$3_$8.log)
    total_lines_ds4=$(wc -l < ./log_bench/$3_$9.log)
    total_lines=$(( $total_lines_ds1 + $total_lines_ds2 + $total_lines_ds3 + $total_lines_ds4 ))
    if [ $total_lines -eq 8 ]; then
      partial_exp_date=$(( $init_exp_date + $6 ));
    else
      partial_exp_date=$(date +%s);
    fi
    echo $(( $partial_exp_date - $init_exp_date ));
    echo $total_lines;
    sleep 10;
  done

} &&

# SYNAS RUN
bench_algorithm=benchmark_SYNAS.py

run_experiments_dataset 0 1 $bench_algorithm 0 0 300 1 2 3

run_experiments_dataset 0 1 $bench_algorithm 4 0 300 5 6 7

#for DSIDX in $(seq 0 2)
#do
#  sh run_shell.sh benchmark_ALIBIC.py 0 0 $DSIDX &
#done;
#sleep 300;
## CarEvaluation Class 0
#for DSIDX in $(seq 0 2)
#do
#  sh run_shell.sh benchmark_ALIBIC.py 1 0 $DSIDX &
#done
## HayesRoth Class 0
#for DSIDX in $(seq 0 79)
#do
#  sh run_shell.sh $1 2 0 $DSIDX &
#done
## Chess Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 3 0 $DSIDX &
#done
## Lymphography Class 0
#for DSIDX in $(seq 0 61)
#do
#  sh run_shell.sh $1 4 0 $DSIDX &
#done
## Nursery Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 5 0 $DSIDX &
#done
## SoybeanSmall Class 0
#for DSIDX in $(seq 0 29)
#do
#  sh run_shell.sh $1 6 0 $DSIDX &
#done
## TicTacToe Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 7 0 $DSIDX &
#done
## BCW Class 0
#for DSIDX in $(seq 0 44)
#do
#  sh run_shell.sh $1 8 0 $DSIDX &
#done
## Ecoli Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 9 0 $DSIDX &
#done
## Iris Class 0
#for DSIDX in $(seq 0 96)
#do
#  sh run_shell.sh $1 10 0 $DSIDX &
#done
## ISOLET Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 11 0 $DSIDX &
#done
## SDD Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 12 0 $DSIDX &
#done
## PBC Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 13 0 $DSIDX &
#done
## CMSC Class 0
#for DSIDX in $(seq 0 3)
#do
#  sh run_shell.sh $1 14 0 $DSIDX &
#done
## MagicGT Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 15 0 $DSIDX &
#done
## Wine Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 16 0 $DSIDX &
#done
## DefaultOfCCC Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 17 0 $DSIDX &
#done
## StudentPerf Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 18 0 $DSIDX &
#done
## Adult Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 19 0 $DSIDX &
#done
## InternetAdv Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 20 0 $DSIDX &
#done
## StatlogGC Class 0
#for DSIDX in $(seq 0 99)
#do
#  sh run_shell.sh $1 21 0 $DSIDX &
#done