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

  if [ $1 -eq 0 ]; then
    LIMIT_0=$(( $(wc -l < ../experiments_data/BalanceScale_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_0=$(( $(wc -l < ../experiments_data/BalanceScale_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_0 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_0 $3 0 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_1=$(( $(wc -l < ../experiments_data/CarEvaluation_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_1=$(( $(wc -l < ../experiments_data/CarEvaluation_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_1 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_1 $3 1 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_2=$(( $(wc -l < ../experiments_data/HayesRoth_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_2=$(( $(wc -l < ../experiments_data/HayesRoth_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_2 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_2 $3 2 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_3=$(( $(wc -l < ../experiments_data/Chess_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_3=$(( $(wc -l < ../experiments_data/Chess_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_3 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_3 $3 3 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_4=$(( $(wc -l < ../experiments_data/Lymphography_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_4=$(( $(wc -l < ../experiments_data/Lymphography_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_4 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_4 $3 4 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_5=$(( $(wc -l < ../experiments_data/Nursery_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_5=$(( $(wc -l < ../experiments_data/Nursery_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_5 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_5 $3 5 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_6=$(( $(wc -l < ../experiments_data/SoybeanSmall_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_6=$(( $(wc -l < ../experiments_data/SoybeanSmall_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_6 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_6 $3 6 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_7=$(( $(wc -l < ../experiments_data/TicTacToe_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_7=$(( $(wc -l < ../experiments_data/TicTacToe_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_7 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_7 $3 7 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_8=$(( $(wc -l < ../experiments_data/BCW_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_8=$(( $(wc -l < ../experiments_data/BCW_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_8 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_8 $3 8 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_9=$(( $(wc -l < ../experiments_data/Ecoli_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_9=$(( $(wc -l < ../experiments_data/Ecoli_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_9 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_9 $3 9 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_10=$(( $(wc -l < ../experiments_data/Iris_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_10=$(( $(wc -l < ../experiments_data/Iris_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_10 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_10 $3 10 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_11=$(( $(wc -l < ../experiments_data/ISOLET_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_11=$(( $(wc -l < ../experiments_data/ISOLET_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_11 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_11 $3 11 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_12=$(( $(wc -l < ../experiments_data/SDD_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_12=$(( $(wc -l < ../experiments_data/SDD_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_12 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_12 $3 12 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_13=$(( $(wc -l < ../experiments_data/PBC_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_13=$(( $(wc -l < ../experiments_data/PBC_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_13 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_13 $3 13 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_14=$(( $(wc -l < ../experiments_data/CMSC_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_14=$(( $(wc -l < ../experiments_data/CMSC_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_14 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_14 $3 14 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_15=$(( $(wc -l < ../experiments_data/MagicGT_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_15=$(( $(wc -l < ../experiments_data/MagicGT_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_15 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_15 $3 15 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_16=$(( $(wc -l < ../experiments_data/Wine_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_16=$(( $(wc -l < ../experiments_data/Wine_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_16 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_16 $3 16 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_17=$(( $(wc -l < ../experiments_data/DefaultOfCCC_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_17=$(( $(wc -l < ../experiments_data/DefaultOfCCC_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_17 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_17 $3 17 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_18=$(( $(wc -l < ../experiments_data/StudentPerf_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_18=$(( $(wc -l < ../experiments_data/StudentPerf_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_18 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_18 $3 18 $1 1800
  fi &&

  if [ $1 -eq 0 ]; then
    LIMIT_19=$(( $(wc -l < ../experiments_data/Adult_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_19=$(( $(wc -l < ../experiments_data/Adult_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_19 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_19 $3 19 $1 1800
  fi &&

#  if [ $1 -eq 0 ]; then
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_0.csv) - 2 ));
#  else
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_1.csv) - 2 ));
#  fi
#
#  run_experiments_dataset $2 0 32 $3 20 $1 1800 &&
#
#  if [ $1 -eq 0 ]; then
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_0.csv) - 2 ));
#  else
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_1.csv) - 2 ));
#  fi
#
#  run_experiments_dataset $2 33 64 $3 20 $1 1800 &&
#
#  if [ $1 -eq 0 ]; then
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_0.csv) - 2 ));
#  else
#    LIMIT_20=$(( $(wc -l < ../experiments_data/InternetAdv_CFDATASET_1.csv) - 2 ));
#  fi
#
#  run_experiments_dataset $2 65 $LIMIT_20 $3 20 $1 1800 &&

  if [ $1 -eq 0 ]; then
    LIMIT_21=$(( $(wc -l < ../experiments_data/StatlogGC_CFDATASET_0.csv) - 2 ));
  else
    LIMIT_21=$(( $(wc -l < ../experiments_data/StatlogGC_CFDATASET_1.csv) - 2 ));
  fi

  if [ $LIMIT_21 -gt 49 ]; then
    run_experiments_dataset $2 50 $LIMIT_21 $3 21 $1 1800
  fi

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

  # SYNAS RUN
  bench_algorithm=benchmark_SYNAS.py
  conda activate SYNAS &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # SEDC RUN
  bench_algorithm=benchmark_SEDC.py
  conda activate SEDC &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # MLEXPLAIN RUN
  bench_algorithm=benchmark_MLEXPLAIN.py
  conda activate MLEXPLAIN &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # MACE RUN
  bench_algorithm=benchmark_MACE.py
  conda activate MACE &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # LORE RUN
  bench_algorithm=benchmark_LORE.py
  conda activate LORE &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # GROWINGSPHERES3 RUN
  bench_algorithm=benchmark_GROWINGSPHERES3.py
  conda activate GROWINGSPHERES &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # GROWINGSPHERES4 RUN
  bench_algorithm=benchmark_GROWINGSPHERES4.py
  conda activate GROWINGSPHERES &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # DICE RUN
  bench_algorithm=benchmark_DiCE.py
  conda activate DICE &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # CADEX RUN
  bench_algorithm=benchmark_CADEX.py
  conda activate CADEX &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # ALIBICNOGRAD RUN
  bench_algorithm=benchmark_ALIBICNOGRAD.py
  conda activate ALIBIC &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm &&

  # ALIBIC RUN
  bench_algorithm=benchmark_ALIBIC.py
  conda activate ALIBIC &&
  run_loop $CAT $RESULT_FOLDER $bench_algorithm

done