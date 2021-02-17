init_date=$(date +%s)
if ! timeout 300 python $1 $2 $3 $4 ; then
  final_date=$(date +%s)
  total_time=$((final_date-init_date))
  echo 'PROCESS TOOK' $total_time 'SECONDS' 'ERROR in '$2 $3 $4
  echo 'PROCESS TOOK' $total_time 'SECONDS' 'ERROR in ' $1 ' AND DSNAME ' $2 'FOR C' $3 'AND ROW' $4 >> ./log_bench/$1.log
else
  echo 'SUCCESS in ' $1 ' AND DSNAME ' $2 >> ./log_bench/$1.log
fi
