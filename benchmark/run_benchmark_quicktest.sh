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
# 1 - Results folder
# 2 - Final row number
# 3 - Framework run algorithm
# 4 - Dataset 1
# 5 - Class
# 6 - Timeout
# 7 - Dataset 2
# 8 - Dataset 3
# 9 - Dataset 4
run_experiments_dataset () {

  LOG_FOLDER=./log_bench/$1

  mkdir -p $LOG_FOLDER

  touch $LOG_FOLDER/$3_$4_$5.log;
  rm $LOG_FOLDER/$3_$4_$5.log;
  touch $LOG_FOLDER/$3_$4_$5.log;
  touch $LOG_FOLDER/$3_$7_$5.log;
  rm $LOG_FOLDER/$3_$7_$5.log;
  touch $LOG_FOLDER/$3_$7_$5.log;
  touch $LOG_FOLDER/$3_$8_$5.log;
  rm $LOG_FOLDER/$3_$8_$5.log;
  touch $LOG_FOLDER/$3_$8_$5.log;
  touch $LOG_FOLDER/$3_$9_$5.log;
  rm $LOG_FOLDER/$3_$9_$5.log;
  touch $LOG_FOLDER/$3_$9_$5.log;
  for DSIDX in $(seq 0 $2)
  do
    sh run_shell_quicktest.sh $3 $4 $5 $DSIDX $1 $LOG_FOLDER &
  done;
  for DSIDX in $(seq 0 $2)
  do
    sh run_shell_quicktest.sh $3 $7 $5 $DSIDX $1 $LOG_FOLDER &
  done;
  for DSIDX in $(seq 0 $2)
  do
    sh run_shell_quicktest.sh $3 $8 $5 $DSIDX $1 $LOG_FOLDER &
  done;
  for DSIDX in $(seq 0 $2)
  do
    sh run_shell_quicktest.sh $3 $9 $5 $DSIDX $1 $LOG_FOLDER &
  done;
  init_exp_date=$(date +%s)
  partial_exp_date=$(date +%s)

  while [ $(( $partial_exp_date - $init_exp_date )) -lt $6 ]
  do
    total_lines_ds1=$(wc -l < $LOG_FOLDER/$3_$4_$5.log)
    total_lines_ds2=$(wc -l < $LOG_FOLDER/$3_$7_$5.log)
    total_lines_ds3=$(wc -l < $LOG_FOLDER/$3_$8_$5.log)
    total_lines_ds4=$(wc -l < $LOG_FOLDER/$3_$9_$5.log)
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

for EXPERIMENT in 0 1 2
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

  # SYNAS RUN
  bench_algorithm=benchmark_SYNAS.py

  conda activate SYNAS &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&


  # SEDC RUN
  bench_algorithm=benchmark_SEDC.py

  conda activate SEDC &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # MLEXPLAIN RUN
  bench_algorithm=benchmark_MLEXPLAIN.py

  conda activate MLEXPLAIN &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # MACE RUN
  bench_algorithm=benchmark_MACE.py

  conda activate MACE &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # LORE RUN
  bench_algorithm=benchmark_LORE.py

  conda activate LORE &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # GROWINGSPHERES3 RUN
  bench_algorithm=benchmark_GROWINGSPHERES3.py

  conda activate GROWINGSPHERES &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # GROWINGSPHERES4 RUN
  bench_algorithm=benchmark_GROWINGSPHERES4.py

  conda activate GROWINGSPHERES &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # DICE RUN
  bench_algorithm=benchmark_DiCE.py

  conda activate DICE &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # CADEX RUN
  bench_algorithm=benchmark_CADEX.py

  conda activate CADEX &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # ALIBICNOGRAD RUN
  bench_algorithm=benchmark_ALIBICNOGRAD.py

  conda activate ALIBIC &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1 &&

  # ALIBIC RUN
  bench_algorithm=benchmark_ALIBIC.py

  conda activate ALIBIC &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 0 $CAT 2400 1 2 3 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 4 $CAT 2400 5 6 7 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 8 $CAT 2400 9 10 11 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 12 $CAT 2400 13 14 15 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 16 $CAT 2400 17 18 19 &&

  run_experiments_dataset $RESULT_FOLDER 1 $bench_algorithm 20 $CAT 2400 21 0 1

done