#python3 main.py -a vgg19_bn --save-dir CIFAR100/vgg19_bn_wd_BlockNSL_inc_0.02std_1e-2alpha_1e-2LR_seed20/ --dataset cifar100 --epochs 150 --lr 1e-2 --alpha 1e-2 --transform-penalty --sigma 0.02 --block --seed 20 --strength 'increase' --sim-loss 'cosine'
#python3 main.py -a vgg19_bn --save-dir CIFAR100/vgg19_bn_wd_BlockNSL_dec_0.02std_1e-2alpha_1e-2LR_seed20/ --dataset cifar100 --epochs 150 --lr 1e-2 --alpha 1e-2 --transform-penalty --sigma 0.02 --block --seed 20 --strength 'decrease' --sim-loss 'cosine'
#python3 main.py -a vgg19_bn --save-dir CIFAR100/vgg19_bn_wd_BlockNSL_const_0.02std_1e-2alpha_1e-2LR_seed20/ --dataset cifar100 --epochs 150 --lr 1e-2 --alpha 1e-2 --transform-penalty --sigma 0.02 --block --seed 20 --strength 'constant' --sim-loss 'cosine'
#python3 main.py -a vgg19_bn --save-dir CIFAR100/vgg19_bn_wd_seed20_1e-2LR/ --epochs 150 --seed 20 --dataset cifar100 --lr 1e-2
#python3 main.py -a vgg19 --save-dir CIFAR100/vgg19_wd_BlockNSL_increase_0.02std_1e-2alpha_1e-2LR_seed10/ --dataset cifar100 --epochs 50 --lr 1e-2 --alpha 1e-2 --transform-penalty --sigma 0.02 --block --seed 10 --strength 'increase' --sim-loss 'cosine'
#python3 main.py -a vgg19 --save-dir vgg19_wd_seed20/ --epochs 70 --seed 20
#python3 main.py -a vgg19 --save-dir vgg19_wd_seed30/ --epochs 70 --seed 30
#python3 main.py -a vgg19 --save-dir vgg19_wd_seed40/ --epochs 70 --seed 40
#python3 main.py -a vgg19 --save-dir vgg19_wd_seed50/ --epochs 70 --seed 50
#python3 main.py -a vgg19 --save-dir vgg19/ --epochs 70 --wd 0 --seed 10
#python3 main.py -a vgg19 --save-dir vgg19_wd_dr/ --epochs 100 --dr
#python3 main.py -a vgg19_bn --save-dir vgg19_bn/ --epochs 70 --wd 0 --seed 10
#python3 main.py -a vgg19_bn --save-dir vgg19_bn_wd/ --epochs 70 --seed 10
