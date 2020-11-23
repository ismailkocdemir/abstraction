#python3 train.py -a resnet18preact --save-dir resnet18_wd/ --epochs 80 --wd 5e-4
#python3 train.py -a resnet18preact --save-dir resnet18_CIFAR100_wd/ --epochs 150 --wd 5e-4 --dataset "cifar100"
#python3 train.py -a resnet18preact --save-dir resnet18_snr_const_alpha1e-2/ --epochs 80 --snr --alpha 1e-2 --strength "constant"
python3 train.py -a resnet18preact --save-dir resnet18_snr_increase_alpha1e-2/ --epochs 80 --snr --alpha 1e-2 --strength "increase"
#python3 train.py -a resnet18preact --save-dir resnet18_CIFAR100_snr_const_alpha1e-2/ --epochs 150 --snr --alpha 1e-2 --strength "constant" --dataset "cifar100"
#python3 train.py -a resnet18preact --save-dir resnet18_CIFAR100_snr_increase_alpha1e-2/ --epochs 150 --snr --alpha 1e-2 --strength "increase" --dataset "cifar100"
#python3 train.py -a vgg19 --save-dir vgg19_snr_const_alpha1e-2/ --epochs 80 --snr --alpha 1e-2 --strength "constant"
#python3 train.py -a vgg19 --save-dir vgg19_snr_decrease_alpha1e-2/ --epochs 80 --snr --alpha 1e-2 --strength 'decrease'
#python3 train.py -a vgg19 --save-dir vgg19_snr_const_alpha1e-2_CIFAR100/ --epochs 150 --snr --alpha 1e-2 --strength "constant" --dataset "cifar100"
#python3 train.py -a vgg19 --save-dir vgg19_snr_decrease_alpha1e-2_CIFAR100/ --epochs 150 --snr --alpha 1e-2 --strength 'decrease' --dataset "cifar100"
#python3 train.py -a vgg19 --save-dir vgg19_wd_5e-4_pytorch/ --epochs 100
#python3 train.py -a vgg19 --save-dir vgg19_ad_a1e-4_l90_singlelayer/ --epochs 100 --ad --alpha 1e-4 --lambd 0.9 --sl --wd 0
#python3 train.py -a vgg19 --save-dir vgg19_ad_a1e-5_l95/ --epochs 100 --ad --alpha 1e-5 --lambd 0.95 --wd 0
#python3 train.py -a vgg19 --save-dir vgg19_wd_5e-4_byme/ --epochs 100 --ad --alpha 5e-4 --lambd 0 --wd 0
#python3 train.py -a vgg19 --save-dir vgg19_ad_a1e-5_l9e-1/ --epochs 100 --ad --alpha 1e-5 --lambd 0.9 --wd 0
#python3 train.py -a vgg19 --save-dir vgg19_ad_a1e-5_l7e-1/ --epochs 100 --ad --alpha 1e-5 --lambd 0.7 --wd 0
#python3 train.py -a vgg19 --save-dir vgg19/ --epochs 100 --wd 0
#python3 train.py -a vgg19 --save-dir checkpoints/vgg19_nl/ --epochs 100 --wd 0 --nl
#python3 train.py -a vgg19 --save-dir checkpoints/vgg19/ --epochs 100 --wd 0
