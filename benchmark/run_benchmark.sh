#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

dataset_order=$(python3 ./scripts/dataset_order_algorithm.py 2>&1)
algorithms_info=$(python3 ./scripts/frameworks_configuration_algorithm.py 2>&1)
start_idx=$1
end_idx=$2

conda init bash&&

# Initial Setup with required Software
echo "Initializing installations"
if [ $(id -u) -eq 0 ]; then
  # Sudo user
  apt update ;
  apt install build-essential -y ;
  apt install wine64 -y ;
else
  # Non-sudo user
  sudo apt update;
  sudo apt install build-essential -y ;
  sudo apt install wine64 -y ;
fi

# Create All Conda Environments Needed
echo "Creating conda environments"
for algo_info in `echo "$algorithms_info" | grep -o -e "[^;]*"`; do
  algo_name="$(cut -d',' -f1 <<<"$algo_info")"
  python_version="$(cut -d',' -f2 <<<"$algo_info")"
  requirements_file_path="$(cut -d',' -f3 <<<"$algo_info")"

  conda create --name $algo_name python=$python_version -y &&
  sleep 10
  conda activate $algo_name &&
  pip install -r $requirements_file_path

  if [ "$algo_name" == "MACE" ]; then
    pysmt-install --z3 --confirm-agreement
  fi

done

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
# 4 - initial index to generate CF
# 5 - final index to generate CF

# LIMIT_X - maximum number of rows
run_loop () {
  echo $dataset_order_algotihm
  for dataset_index in `echo "$dataset_order" | grep -o -e "[^;]*"`; do
    dsIdx="$(cut -d',' -f1 <<<"$dataset_index")"
    dsName="$(cut -d',' -f2 <<<"$dataset_index")"

    dsLimitRows=$(( $(wc -l < "../dataset_data/experiments_data/"$dsName"_CFDATASET_"$1".csv") - 2 ));

    if [ $5 -ge $dsLimitRows ]; then
      UPPER_LIMIT=$dsLimitRows
    else
      UPPER_LIMIT=$5
    fi

    if [ $4 -le $dsLimitRows ]; then
      run_experiments_dataset $2 $4 $UPPER_LIMIT $3 $dsIdx $1 1800;
    fi

  done

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

  for algo_info in `echo "$algorithms_info" | grep -o -e "[^;]*"`; do
    algo_name="$(cut -d',' -f1 <<<"$algo_info")"
    bench_algorithm="$(cut -d',' -f4 <<<"$algo_info")"

    # ALGORITHM RUN
    conda activate $algo_name &&
    run_loop $CAT $RESULT_FOLDER $bench_algorithm $start_idx $end_idx

  done

done