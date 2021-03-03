# Deepest Season9 challenge Q1

## Data & task
Garbage img data Classification (https://www.kaggle.com/asdasdasasdas/garbage-classification)

## Implementations
1. Augmentations
 - CutMix
 - Mixup
 
 Belows are implemented using albumentation (https://albumentations.readthedocs.io/)
 - HorizontalFlip 
 - RandomRotate90
 - VerticalFlip
 - MotionBlur
 - OpticalDistortion
 - GaussNoise
 
2. Snapshot Ensemble (https://arxiv.org/pdf/1704.00109.pdf)
   - Train N networks with 1 training
   
3. Used MobileNetv3 from (https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py)

## requirements
```shell script
pip install torch
```

## usage
First, fill your neptune (https://neptune.ai) data to enjoy powerful and beautiful logging, 
then

- To train :
```python
    python ensemble.py --save_dir=without_mixup --no_use_mix  # 91.29 % test acc
```
```python
    python ensemble.py --save_dir=with_mixup # expects 88.13 % test acc
```

- To enjoy the trained net:
First, untar the weight files
```shell script
    tar -xzf without_mixup
    #or
    tar -xzf with_mixup
```

test with ensemble
```python
    python play.py --save_dir=without_mixup --ensemble
``` 
test with single best network
```python
    python play.py --save_dir=without_mixup --single
```    

## Results
https://ui.neptune.ai/jeffrey/deepest-season9/e/DEEP-47/charts
