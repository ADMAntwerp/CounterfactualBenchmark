for DSIDX in $(seq $1 $2)
do
  sh run_shell.sh $3 $4 $5 $DSIDX &
done
