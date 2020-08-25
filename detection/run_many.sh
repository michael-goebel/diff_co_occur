


nohup bash run_one_gpu.sh vgg16 0 > logs/vgg16.txt &
nohup bash run_one_gpu.sh resnet18 1 > logs/resnet18.txt &
nohup bash run_one_gpu.sh resnet152 2 > logs/resnet152.txt &
nohup bash run_one_gpu.sh resnext50 3 > logs/resnext50.txt &
nohup bash run_one_gpu.sh inception_v3 4 > logs/inception_v3.txt &






