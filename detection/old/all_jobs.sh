
nohup python train_models2.py co_occur resnet18 all_pgd 0 &> logs2/co_occur_resnet18_all_pdg.txt &
nohup python train_models2.py co_occur resnet18 all_adv 1 &> logs2/co_occur_resnet18_all_adv.txt &
nohup python train_models2.py dft resnet18 none 2 &> logs2/dft_resnet18_none.txt &
nohup python train_models2.py dft resnet18 all_cc 3 &> logs2/dft_resnet18_all_cc.txt &
nohup python train_models2.py dft resnet18 all_dft 4 &> logs2/dft_resnet18_all_dft.txt &
nohup python train_models2.py dft resnet18 all_pgd 5 &> logs2/dft_resnet18_all_pgd.txt &


nohup python train_models2.py dft resnet18 all_adv 0 &> logs2/dft_resnet18_all_adv.txt &
nohup python train_models2.py direct resnet18 none 1 &> logs2/direct_resnet18_none.txt &
nohup python train_models2.py direct resnet18 all_cc 2 &> logs2/direct_resnet18_all_cc.txt &
nohup python train_models2.py direct resnet18 all_dft 3 &> logs2/direct_resnet18_all_dft.txt &
nohup python train_models2.py direct resnet18 all_pgd 4 &> logs2/direct_resnet18_all_pgd.txt &
nohup python train_models2.py direct resnet18 all_adv 5 &> logs2/direct_resnet18_all_adv.txt &


nohup python train_models2.py co_occur mobilenet none 0 &> logs2/co_occur_mobilenet_none.txt &
nohup python train_models2.py dft mobilenet none 1 &> logs2/dft_mobilenet_none.txt &
nohup python train_models2.py direct mobilenet none 2 &> logs2/direct_mobilenet_none.txt &

nohup python train_models2.py co_occur mobilenet all_adv 3 &> logs2/co_occur_mobilenet_all_adv.txt &
nohup python train_models2.py dft mobilenet all_adv 4 &> logs2/dft_mobilenet_all_adv.txt &
nohup python train_models2.py direct mobilenet all_adv 5 &> logs2/direct_mobilenet_all_adv.txt &






