#!/bin/bash

lambda=3.0

n_gpu=5
n_per_gpu=2


params_list="
0,933,train
933,1867,train
1867,2800,train
2800,3733,train
3733,4667,train
4667,5600,train
5600,6533,train
6533,7467,train
7467,8400,train
8400,9333,train
9333,10267,train
10267,11200,train
11200,12133,train
12133,13067,train
13067,14000,train
14000,14933,train
14933,15867,train
15867,16800,train
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











