python3 representation_analysis.py --rm spectral-analysis -a vgg19 -s 5 --mode weight
#python3 representation_analysis.py --rm spectral-analysis -a vgg19 -s 5 --mode activation
python3 representation_analysis.py --rm spectral-analysis -a vgg19_wd -s 5 --mode weight
#python3 representation_analysis.py --rm spectral-analysis -a vgg19_wd -s 5 --mode activation
python3 representation_analysis.py --rm spectral-analysis -a vgg19_bn -s 5 --mode weight
#python3 representation_analysis.py --rm spectral-analysis -a vgg19_bn -s 5 --mode activation
python3 representation_analysis.py --rm spectral-analysis -a vgg19_bn_wd -s 5 --mode weight
#python3 representation_analysis.py --rm spectral-analysis -a vgg19_bn_wd -s 5 --mode activation
