#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda init bash&&

# Initial Setup with required Software
echo "Initializing installations"
if [ $(id -u) -eq 0 ]; then
  # Sudo user
  apt install build-essential -y ;
  apt install wine64 -y ;
else
  # Non-sudo user
  sudo apt install build-essential -y ;
  sudo apt install wine64 -y ;
fi

# Create All Conda Environments Needed
echo "Creating conda environments"
conda create --name ALIBIC python=3.7 -y &&
sleep 10
conda activate ALIBIC &&
pip install -r ../framework_requirements/alibic_requirements.txt &&

conda create --name CADEX python=3.6 -y &&
sleep 10
conda activate CADEX &&
pip install -r ../framework_requirements/cadex_requirements.txt &&

conda create --name DICE python=3.7 -y &&
sleep 10
conda activate DICE &&
pip install -r ../framework_requirements/dice_requirements.txt &&

conda create --name GROWINGSPHERES python=3.6 -y &&
sleep 10
conda activate GROWINGSPHERES &&
pip install -r ../framework_requirements/growingspheres_requirements.txt &&

conda create --name LORE python=3.7 -y &&
sleep 10
conda activate LORE &&
pip install -r ../framework_requirements/lore_requirements.txt &&

conda create --name MACE python=3.6 -y &&
sleep 10
conda activate MACE &&
pip install -r ../framework_requirements/mace_requirements.txt &&
pysmt-install --z3 --confirm-agreement &&


conda create --name MLEXPLAIN python=3.7 -y &&
sleep 10
conda activate MLEXPLAIN &&
pip install -r ../framework_requirements/mlexplain_requirements.txt &&

conda create --name SEDC python=3.7 -y &&
sleep 10
conda activate SEDC &&
pip install -r ../framework_requirements/sedc_requirements.txt &&

conda create --name SYNAS python=3.7 -y &&
sleep 10
conda activate SYNAS &&
pip install -r ../framework_requirements/synas_requirements.txt &&

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
    sh run_shell_quicktest.sh $3 $4 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell_quicktest.sh $3 $7 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell_quicktest.sh $3 $8 $5 $DSIDX &
  done;
  for DSIDX in $(seq $1 $2)
  do
    sh run_shell_quicktest.sh $3 $9 $5 $DSIDX &
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

conda activate SYNAS &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# SEDC RUN
bench_algorithm=benchmark_SEDC.py

conda activate SEDC &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# MLEXPLAIN RUN
bench_algorithm=benchmark_MLEXPLAIN.py

conda activate MLEXPLAIN &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# MACE RUN
bench_algorithm=benchmark_MACE.py

conda activate MACE &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# LORE RUN
bench_algorithm=benchmark_LORE.py

conda activate LORE &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# GROWINGSPHERES3 RUN
bench_algorithm=benchmark_GROWINGSPHERES3.py

conda activate GROWINGSPHERES &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# GROWINGSPHERES4 RUN
bench_algorithm=benchmark_GROWINGSPHERES4.py

conda activate GROWINGSPHERES &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# DICE RUN
bench_algorithm=benchmark_DiCE.py

conda activate DICE &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# CADEX RUN
bench_algorithm=benchmark_CADEX.py

conda activate CADEX &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# ALIBICNOGRAD RUN
bench_algorithm=benchmark_ALIBICNOGRAD.py

conda activate ALIBIC &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1 &&

# ALIBIC RUN
bench_algorithm=benchmark_ALIBIC.py

conda activate ALIBIC &&

run_experiments_dataset 0 2 $bench_algorithm 0 0 2400 1 2 3 &&

run_experiments_dataset 0 2 $bench_algorithm 4 0 2400 5 6 7 &&

run_experiments_dataset 0 2 $bench_algorithm 8 0 2400 9 10 11 &&

run_experiments_dataset 0 2 $bench_algorithm 12 0 2400 13 14 15 &&

run_experiments_dataset 0 2 $bench_algorithm 16 0 2400 17 18 19 &&

run_experiments_dataset 0 2 $bench_algorithm 20 0 2400 21 0 1