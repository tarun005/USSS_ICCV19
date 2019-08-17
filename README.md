# USSS_ICCV19

Code for Universal Semi Supervised Semantic Segmentation accepted to ICCV 2019, full Paper available at https://arxiv.org/abs/1811.10323. 

## How to run

### Requirements

Python >= 2.6  
PyTorch >= 1.0.0

### Datasets
Cityscapes: https://www.cityscapes-dataset.com/  
IDD: https://idd.insaan.iiit.ac.in/

### Command
``python segment.py --basedir <basedir> --lr 0.001 --num-epochs 200 --batch-size 8 --savedir <savedir> --datasets <D1> [<D2> ..]  --num-samples <N> --alpha 0 --beta 0 --resnet <resnet_v> --model drnet``
