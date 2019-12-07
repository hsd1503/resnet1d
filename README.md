# Introduction

This is a pytorch implementation of ResNet [0] and ResNeXt [1] on one-dimensional data classification. 

# Usage

```
# test on synthetic data
python test_synthetic.py

# test on PhysioNet/CinC Challenge 2017 data
python test_physionet.py
```

# Requirements

python 3.7.5, pytorch 1.2.0


# Applications

## ECG Classification (PhysioNet/CinC Challenge 2017)

Dataset: "AF Classification from a short single lead ECG recording". Data can be found at https://archive.physionet.org/challenge/2017/#challenge-data Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv

The model has been used in one of the First place solution (F1=0.83) [2, 3]. The original tensorflow (tflearn) version can be found at https://github.com/hsd1503/ENCASE. 



# References

[0] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR 2016 [paper](https://arxiv.org/abs/1512.03385)

[1] Saining Xie, Ross B. Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He: Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017 [paper](https://arxiv.org/abs/1611.05431)

[2] Shenda Hong, Meng Wu, Yuxi Zhou, Qingyun Wang, Junyuan Shang, Hongyan Li, Junqing Xie. ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks. Computing in Cardiology (CinC) Conference 2017 [paper](http://www.cinc.org/archives/2017/pdf/178-245.pdf)

[3] Shenda Hong, Yuxi Zhou, Meng Wu, Qingyun Wang, Junyuan Shang, Hongyan Li and Junqing Xie. Combining Deep Neural Networks and Engineered Features for Cardiac Arrhythmia Detection from ECG Recordings. Physiological Measurement 2019 [paper](https://www.ncbi.nlm.nih.gov/pubmed/30943458)

[4] Yuxi Zhou, Shenda Hong, Meng Wu, Junyuan Shang, Qingyun Wang, Junqing Xie, Hongyan Li. K-margin-based Residual-convolution-recurrent Neural Network for Atrial Fibrillation Detection. International Joint Conference on Artificial Intelligence (IJCAI) 2019 [paper](https://www.ijcai.org/proceedings/2019/0839.pdf)
