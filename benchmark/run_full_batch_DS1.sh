# Run all class 1 Datasets
# BalanceScale Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 0 1 $DSIDX &
done
# CarEvaluation Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 1 1 $DSIDX &
done
# HayesRoth Class 1
for DSIDX in $(seq 0 36)
do
  sh run_shell.sh $1 2 1 $DSIDX &
done
# Chess Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 3 1 $DSIDX &
done
# Lymphography Class 1
for DSIDX in $(seq 0 77)
do
  sh run_shell.sh $1 4 1 $DSIDX &
done
# Nursery Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 5 1 $DSIDX &
done
# SoybeanSmall Class 1
for DSIDX in $(seq 0 16)
do
  sh run_shell.sh $1 6 1 $DSIDX &
done
# TicTacToe Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 7 1 $DSIDX &
done
# BCW Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 8 1 $DSIDX &
done
# Ecoli Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 9 1 $DSIDX &
done
# Iris Class 1
for DSIDX in $(seq 0 49)
do
  sh run_shell.sh $1 10 1 $DSIDX &
done
# ISOLET Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 11 1 $DSIDX &
done
# SDD Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 12 1 $DSIDX &
done
# PBC Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 13 1 $DSIDX &
done
# CMSC Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 14 1 $DSIDX &
done
# MagicGT Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 15 1 $DSIDX &
done
# Wine Class 1
for DSIDX in $(seq 0 69)
do
  sh run_shell.sh $1 16 1 $DSIDX &
done
# DefaultOfCCC Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 17 1 $DSIDX &
done
# StudentPerf Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 18 1 $DSIDX &
done
# Adult Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 19 1 $DSIDX &
done
# InternetAdv Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 20 1 $DSIDX &
done
# StatlogGC Class 1
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 21 1 $DSIDX &
done