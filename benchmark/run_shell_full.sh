init_date=$(date +%s)
if ! timeout 1800 python $1 $2 $3 $4 $5 ; then
  final_date=$(date +%s)
  total_time=$((final_date-init_date))
  echo 'PROCESS TOOK' $total_time 'SECONDS' 'ERROR in '$2 $3 $4
  echo 'PROCESS TOOK' $total_time 'SECONDS' 'ERROR in ' $1 ' AND DSNAME ' $2 'FOR C' $3 'AND ROW' $4 >> $6/$7_$8_$1_$2_$3.log
else
  final_date=$(date +%s)
  total_time=$((final_date-init_date))
  echo 'PROCESS TOOK' $total_time 'SECONDS SUCCESS in ' $1 ' AND DSNAME ' $2 >> $6/$1_$2_$3.log
fi
