k=0
for i in 0,10.0 1,3.0 2,1.0 3,0.0;
do
	IFS=','; set $i; gpu=$1; lambda=$2
	#for j in 0,75 75,150 150,225 225,300;
	for j in 0,150 150,300;
	do
		IFS=','; set $j; start=$1; end=$2
		#echo "$lambda $gpu $start $end 300"

		nohup python run_diff_lambda.py $lambda $gpu $start $end 300 > log${k}.txt &
		k=$(($k+1))
		#echo $k

	done	

done



#nohup python run_diff_lambda.py 3.0 0 0 100 400 > log1.txt &
#nohup python run_diff_lambda.py 3.0 0 100 200 400 > log2.txt &
#nohup python run_diff_lambda.py 3.0 0 200 300 400 > log3.txt &
#nohup python run_diff_lambda.py 3.0 0 300 400 400 > log3.txt &
