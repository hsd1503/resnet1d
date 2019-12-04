# Introduction

This is a pytorch implementation of ResNet [0] on one-dimensional data classification. 

# Usage

python test_synthetic.py to test on synthetic data

# Requirements

python 3.7.5, pytorch 1.2.0


# Applications

## ECG Classification (PhysioNet/CinC Challenge 2017)

Dataset: "AF Classification from a short single lead ECG recording". Data can be found at https://archive.physionet.org/challenge/2017/#challenge-data Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv

The model has been used in one of the First place solution (F1=0.83) [1, 2]. The original tensorflow (tflearn) version can be found at https://github.com/hsd1503/ENCASE. 



# References

[0] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[1] Shenda Hong, Meng Wu, Yuxi Zhou, Qingyun Wang, Junyuan Shang, Hongyan Li, Junqing Xie. ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks. Computing in Cardiology (CinC) Conference 2017

[2] Shenda Hong, Yuxi Zhou, Meng Wu, Qingyun Wang, Junyuan Shang, Hongyan Li and Junqing Xie. Combining Deep Neural Networks and Engineered Features for Cardiac Arrhythmia Detection from ECG Recordings. Physiological Measurement 2019

[3] Yuxi Zhou, Shenda Hong, Meng Wu, Junyuan Shang, Qingyun Wang, Junqing Xie, Hongyan Li. K-margin-based Residual-convolution-recurrent Neural Network for Atrial Fibrillation Detection. International Joint Conference on Artificial Intelligence (IJCAI) 2019
