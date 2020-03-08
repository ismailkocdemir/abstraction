python3 representation.py -a resnet18  --max-epoch 50 -m RV
python3 representation.py -a resnet18  --max-epoch 50 -m pwcca
python3 representation.py -a vgg19  --max-epoch 50 -m RV
python3 representation.py -a vgg19 --max-epoch 50 -m pwcca
python3 representation.py -a vgg19_bn --max-epoch 50 -m RV
python3 representation.py -a vgg19_bn  --max-epoch 50 -m pwcca

