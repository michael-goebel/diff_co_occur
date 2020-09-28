python train_models_w_jpeg.py dft resnet18 all_adv 6 &> logs_jpeg/dft_resnet18_all_adv.txt
python train_models_w_jpeg.py direct resnet18 none 6 &> logs_jpeg/direct_resnet18_none.txt
python train_models_w_jpeg.py direct resnet18 all_cc 6 &> logs_jpeg/direct_resnet18_all_cc.txt
python train_models_w_jpeg.py direct resnet18 all_dft 6 &> logs_jpeg/direct_resnet18_all_dft.txt
python train_models_w_jpeg.py direct resnet18 all_pgd 6 &> logs_jpeg/direct_resnet18_all_pgd.txt
python train_models_w_jpeg.py direct resnet18 all_adv 6 &> logs_jpeg/direct_resnet18_all_adv.txt
