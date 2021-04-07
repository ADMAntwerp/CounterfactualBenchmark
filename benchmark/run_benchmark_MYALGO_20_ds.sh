#!/bin/bash
# Run benchmark on very large dataset, requiring more memory resources
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
conda create --name MYALGO python=3.7 -y &&
sleep 10
conda activate MYALGO &&
pip install -r ../framework_requirements/myalgo_requirements.txt &&

# Run all class 0 Datasets
# 1 - Results folder
# 2 - Initial row number
# 3 - Final row number
# 4 - Framework run algorithm
# 5 - Dataset 1
# 6 - Class
# 7 - Timeout
run_experiments_dataset () {

  LOG_FOLDER=./log_bench/$1

  mkdir -p $LOG_FOLDER

  touch $LOG_FOLDER/$2_$3_$4_$5_$6.log;
  rm $LOG_FOLDER/$2_$3_$4_$5_$6.log;
  touch $LOG_FOLDER/$2_$3_$4_$5_$6.log;

  for DSIDX in $(seq $2 $3)
  do
    sh run_shell_full.sh $4 $5 $6 $DSIDX $1 $LOG_FOLDER $2 $3 &
  done;

  init_exp_date=$(date +%s)
  partial_exp_date=$(date +%s)

  while [ $(( $partial_exp_date - $init_exp_date )) -lt $7 ]
  do
    total_lines=$(wc -l < $LOG_FOLDER/$2_$3_$4_$5_$6.log)
    if [ $total_lines -eq $(( $3 + 1 - $2 )) ]; then
      partial_exp_date=$(( $init_exp_date + $7 ));
    else
      partial_exp_date=$(date +%s);
    fi
    echo $(( $partial_exp_date - $init_exp_date ));
    echo $total_lines;
    sleep 10;
  done

} &&

# Run Experiment loop
# 1 - CAT
# 2 - RESULT_FOLDER
# 3 - bench_algorithm
run_loop () {
  # Only run the 20 dataset

  run_experiments_dataset $2 0 32 $3 20 $1 1800 &&

  run_experiments_dataset $2 33 64 $3 20 $1 1800 &&

  run_experiments_dataset $2 65 99 $3 20 $1 1800

} &&

for EXPERIMENT in 0 1 2 3
do
  if [ $EXPERIMENT -eq 0 ]; then
    RESULT_FOLDER=results
    CAT=0
  fi
  if [ $EXPERIMENT -eq 1 ]; then
    RESULT_FOLDER=results
    CAT=1
  fi
  if [ $EXPERIMENT -eq 2 ]; then
    RESULT_FOLDER=replication
    CAT=0
  fi
  if [ $EXPERIMENT -eq 3 ]; then
    RESULT_FOLDER=replication
    CAT=1
  fi

  # MYALGO RUN
  bench_algorithm=benchmark_MYALGO.py
  conda activate MYALGO &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm

done