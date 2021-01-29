if ! python $1 $2 ; then
  echo 'ERROR in '$2
  echo 'ERROR in ' $1 ' AND DSNAME ' $2 >> ./log_bench/$1.log
else
  echo 'SUCCESS in ' $1 ' AND DSNAME ' $2 >> ./log_bench/$1.log
fi
