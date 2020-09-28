python train_models_w_jpeg.py co_occur resnet18 all_pgd 5 &> logs_jpeg/co_occur_resnet18_all_pdg.txt
python train_models_w_jpeg.py co_occur resnet18 all_adv 5 &> logs_jpeg/co_occur_resnet18_all_adv.txt
python train_models_w_jpeg.py dft resnet18 none 5 &> logs_jpeg/dft_resnet18_none.txt
python train_models_w_jpeg.py dft resnet18 all_cc 5 &> logs_jpeg/dft_resnet18_all_cc.txt
python train_models_w_jpeg.py dft resnet18 all_dft 5 &> logs_jpeg/dft_resnet18_all_dft.txt
python train_models_w_jpeg.py dft resnet18 all_pgd 5 &> logs_jpeg/dft_resnet18_all_pgd.txt
