# USSS_ICCV19

Code for Universal Semi Supervised Semantic Segmentation accepted to ICCV 2019, full Paper available at https://arxiv.org/abs/1811.10323. 



### Requirements

Python >= 2.6  
PyTorch >= 1.0.0 <br>
The ImageNet pretrained models are downloaded from the repository at https://github.com/fyu/drn. 

### Datasets
Cityscapes: https://www.cityscapes-dataset.com/  
IDD: https://idd.insaan.iiit.ac.in/

### How to run
``python segment.py --basedir <basedir> --lr 0.001 --num-epochs 200 --batch-size 8 --savedir <savedir> --datasets <D1> [<D2> ..]  --num-samples <N> --alpha 0 --beta 0 --resnet <resnet_v> --model drnet``

## Acknowledgements

Part of the code is heavily borrowed from the official code release of Dilated Residual Networks (https://github.com/fyu/drn) and IDD Dataset (https://github.com/AutoNUE/public-code).
