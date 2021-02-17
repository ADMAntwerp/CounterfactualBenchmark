# Run all class 0 Datasets
# BalanceScale Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 0 0 $DSIDX &
done
# CarEvaluation Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 1 0 $DSIDX &
done
# HayesRoth Class 0
for DSIDX in $(seq 0 79)
do
  sh run_shell.sh $1 2 0 $DSIDX &
done
# Chess Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 3 0 $DSIDX &
done
# Lymphography Class 0
for DSIDX in $(seq 0 61)
do
  sh run_shell.sh $1 4 0 $DSIDX &
done
# Nursery Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 5 0 $DSIDX &
done
# SoybeanSmall Class 0
for DSIDX in $(seq 0 29)
do
  sh run_shell.sh $1 6 0 $DSIDX &
done
# TicTacToe Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 7 0 $DSIDX &
done
# BCW Class 0
for DSIDX in $(seq 0 44)
do
  sh run_shell.sh $1 8 0 $DSIDX &
done
# Ecoli Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 9 0 $DSIDX &
done
# Iris Class 0
for DSIDX in $(seq 0 96)
do
  sh run_shell.sh $1 10 0 $DSIDX &
done
# ISOLET Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 11 0 $DSIDX &
done
# SDD Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 12 0 $DSIDX &
done
# PBC Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 13 0 $DSIDX &
done
# CMSC Class 0
for DSIDX in $(seq 0 3)
do
  sh run_shell.sh $1 14 0 $DSIDX &
done
# MagicGT Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 15 0 $DSIDX &
done
# Wine Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 16 0 $DSIDX &
done
# DefaultOfCCC Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 17 0 $DSIDX &
done
# StudentPerf Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 18 0 $DSIDX &
done
# Adult Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 19 0 $DSIDX &
done
# InternetAdv Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 20 0 $DSIDX &
done
# StatlogGC Class 0
for DSIDX in $(seq 0 99)
do
  sh run_shell.sh $1 21 0 $DSIDX &
done