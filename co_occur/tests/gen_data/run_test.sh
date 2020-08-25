
lambda=0.0

python run_diff_lambda.py 0 0.0 0 5 0 1000 train &> log1.txt &
python run_diff_lambda.py 0 0.0 5 10 0 1000 train &> log2.txt &
#nohup python run_diff_lambda.py 0 0.0 10 15 0 1000 train > log3.txt &
#nohup python run_diff_lambda.py 0 0.0 15 20 0 1000 train > log4.txt &


# $start $end $b_start $b_end $mode > $log_file &





