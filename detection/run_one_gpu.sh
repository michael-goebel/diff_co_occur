
source /home/mgoebel/summer_2020/py3/bin/activate

python train_models1.py co_occur $1 $2 $3 &> logs/co_occur_${1}_${2}.txt
python train_models1.py dft $1 $2 $3 &> logs/dft_${1}_${2}.txt
python train_models1.py direct $1 $2 $3 &> logs/direct_${1}_${2}.txt



