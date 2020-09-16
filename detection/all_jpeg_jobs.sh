Z
python train_models_w_jpeg.py co_occur resnet18 none 0 &> logs_jpeg/co_occur_resnet18_none.txt
python train_models_w_jpeg.py co_occur resnet18 cc_0.0 0 &> logs_jpeg/co_occur_resnet18_cc_0.0.txt
python train_models_w_jpeg.py co_occur resnet18 cc_3.0 0 &> logs_jpeg/co_occur_resnet18_cc_3.0.txt
python train_models_w_jpeg.py co_occur resnet18 cc_10.0 0 &> logs_jpeg/co_occur_resnet18_cc_10.0.txt
python train_models_w_jpeg.py co_occur resnet18 all_cc 0 &> logs_jpeg/co_occur_resnet18_all_cc.txt
python train_models_w_jpeg.py co_occur resnet18 all_dft 0 &> logs_jpeg/co_occur_resnet18_all_dft.txt
python train_models_w_jpeg.py co_occur resnet18 all_pgd 0 &> logs_jpeg/co_occur_resnet18_all_pdg.txt
python train_models_w_jpeg.py co_occur resnet18 all_adv 1 &> logs_jpeg/co_occur_resnet18_all_adv.txt

python train_models_w_jpeg.py dft resnet18 none 2 &> logs_jpeg/dft_resnet18_none.txt
python train_models_w_jpeg.py dft resnet18 all_cc 3 &> logs_jpeg/dft_resnet18_all_cc.txt
python train_models_w_jpeg.py dft resnet18 all_dft 4 &> logs_jpeg/dft_resnet18_all_dft.txt
python train_models_w_jpeg.py dft resnet18 all_pgd 5 &> logs_jpeg/dft_resnet18_all_pgd.txt
python train_models_w_jpeg.py dft resnet18 all_adv 0 &> logs_jpeg/dft_resnet18_all_adv.txt

python train_models_w_jpeg.py direct resnet18 none 1 &> logs_jpeg/direct_resnet18_none.txt
python train_models_w_jpeg.py direct resnet18 all_cc 2 &> logs_jpeg/direct_resnet18_all_cc.txt
python train_models_w_jpeg.py direct resnet18 all_dft 3 &> logs_jpeg/direct_resnet18_all_dft.txt
python train_models_w_jpeg.py direct resnet18 all_pgd 4 &> logs_jpeg/direct_resnet18_all_pgd.txt
python train_models_w_jpeg.py direct resnet18 all_adv 5 &> logs_jpeg/direct_resnet18_all_adv.txt

python train_models_w_jpeg.py co_occur mobilenet none 0 &> logs_jpeg/co_occur_mobilenet_none.txt
python train_models_w_jpeg.py dft mobilenet none 1 &> logs_jpeg/dft_mobilenet_none.txt
python train_models_w_jpeg.py direct mobilenet none 2 &> logs_jpeg/direct_mobilenet_none.txt
python train_models_w_jpeg.py co_occur mobilenet all_adv 3 &> logs_jpeg/co_occur_mobilenet_all_adv.txt
python train_models_w_jpeg.py dft mobilenet all_adv 4 &> logs_jpeg/dft_mobilenet_all_adv.txt
python train_models_w_jpeg.py direct mobilenet all_adv 5 &> logs_jpeg/direct_mobilenet_all_adv.txt




