#!/bin/bash

lambda=3.0

n_gpu=5
n_per_gpu=2


params_list="
0,901,val
901,1802,val
1802,2703,val
2703,3604,val
0,899,test
899,1798,test
1798,2697,test
2697,3596,test
"


log_dir="logs"

mkdir -p $log_dir


for p in $params_list
do
	#echo $p
	IFS=','; set $p; b_start=$1; b_end=$2; mode=$3
	n_split=$((${n_gpu}*${n_per_gpu}))
	L=$((${b_end}-${b_start}))
	#echo $L
	
	step=$((${L}/${n_split}))
	#echo $step

	echo "new batch"

	i=0

	for ((gpu = 0 ; gpu < $n_gpu ; gpu++));
	do
		for ((group = 0 ; group < $n_per_gpu ; group++));
		do
			#echo "gpu and group" $gpu $group
	
			start=$((${step}*${i}))
			end=$((${step}*(${i}+1)))

			i=$((${i}+1))

			if (( $i == $n_split ))
			then
				end=$L
			fi

			#echo  $gpu $lambda $start $end $b_start $b_end $mode	
			log_file=${log_dir}/${lambda}_${start}_${end}_${b_start}_${b_end}_${mode}.txt
			python run_diff_lambda.py $gpu $lambda $start $end $b_start $b_end $mode > $log_file &
			
			pids[${i}]=$!

		done	
				

	done

	for pid in ${pids[*]}; do
		wait $pid
	done


done	











