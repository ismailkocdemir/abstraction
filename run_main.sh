#python3 main.py -a vgg19 --save-dir checkpoints/vgg19_nl/ --epochs 100 --wd 0 --nl
#python3 main.py -a vgg19 --save-dir checkpoints/vgg19/ --epochs 100 --wd 0
#python3 main.py -a vgg19 --save-dir checkpoints/vgg19_wd_dr/ --epochs 100 --dr
#python3 main.py -a vgg19 --save-dir checkpoints/vgg19_wd/ --epochs 100
#python3 main.py -a vgg19_bn --save-dir vgg19_bn/ --epochs 100 --wd 0
python3 main.py -a vgg19_bn --save-dir vgg19_bn_wd/ --epochs 100
#python3 main.py -a vgg19 --save-dir checkpoints/vgg19_dr/ --epochs 100 --dr 
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_wd_dr/ --epochs 100 --dr
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_wd/ --epochs 100
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_dr/ --epochs 100 --dr 
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn/ --epochs 100 --wd 0
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_wd_dr/ --epochs 100 --dr --optimizer adam
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_wd/ --epochs 100 --optimizer adam
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn_dr/ --epochs 100 --dr --optimizer adam
#python3 main.py -a vgg19_bn --save-dir checkpoints/vgg19_bn/ --epochs 100 --wd 0 --optimizer adam
#python3 main.py -a resnet18preact --save-dir checkpoints/resnet18preact/ --epochs 100
#python3 main.py -a resnet18preact --save-dir checkpoints/resnet18preact_adam/ --epochs 100 --optimizer adam
#python3 main.py -a resnet18preact_bn --save-dir checkpoints/resnet18preact_bn --epochs 100
#python3 main.py -a resnet18preact_bn --save-dir checkpoints/resnet18preact_bn_adam --epochs 100 --optimizer adam
