python3 main.py -a vgg19 --resume checkpoints/vgg19/checkpoint_97.tar --evaluate
python3 main.py -a vgg19_bn --resume checkpoints/vgg19_bn/checkpoint_94.tar --evaluate
python3 main.py -a vgg19_bn --resume checkpoints/vgg19_bn_adam/checkpoint_96.tar --evaluate
python3 main.py -a resnet18preact --resume checkpoints/resnet18preact/checkpoint_93.tar --evaluate
python3 main.py -a resnet18preact --resume checkpoints/resnet18preact_adam/checkpoint_93.tar --evaluate
python3 main.py -a resnet18preact_bn --resume checkpoints/resnet18preact_bn/checkpoint_93.tar --evaluate
python3 main.py -a resnet18preact_bn --resume checkpoints/resnet18preact_bn_adam/checkpoint_94.tar --evaluate
