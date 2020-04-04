In construction ...

# Introduction

This is a PyTorch implementation of ResNet [0] and ResNeXt [1] on one-dimensional data classification, with applications on ECG Classification. 

# Usage

```
# test on synthetic data, no data download required
python test_synthetic.py

# test on PhysioNet/CinC Challenge 2017 data
# need prepare data first, or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
# Please see comment in code for details
python test_physionet.py

# training or serving model with Ray [5], need install Ray first: https://github.com/ray-project/ray
# Please see comment in code for details
python test_ray.py
```

model_detail/ shows model architectures

# Requirements

Required: Python 3.7.5, PyTorch 1.2.0, torchsummary

Optional: Ray 0.8.0

# Applications: ECG Classification (PhysioNet/CinC Challenge 2017)

Dataset: "AF Classification from a short single lead ECG recording". Data can be found at https://archive.physionet.org/challenge/2017/#challenge-data Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv , or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf .

This repository also contains data preprocessing code, please see util.py for details.

The model has been used in our previous work [2,3] for deep feature extraction, which won one of the First place (F1=0.83) of this Challenge. The original tensorflow (tflearn) version can be found at https://github.com/hsd1503/ENCASE (If you use this code in your work, please cite our papers). 

# References

[0] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR 2016 [paper](https://arxiv.org/abs/1512.03385)

[1] Saining Xie, Ross B. Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He: Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017 [paper](https://arxiv.org/abs/1611.05431)

[2] Shenda Hong, Meng Wu, Yuxi Zhou, Qingyun Wang, Junyuan Shang, Hongyan Li, Junqing Xie. ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks. Computing in Cardiology (CinC) Conference 2017 [paper](http://www.cinc.org/archives/2017/pdf/178-245.pdf)

[3] Shenda Hong, Yuxi Zhou, Meng Wu, Qingyun Wang, Junyuan Shang, Hongyan Li and Junqing Xie. Combining Deep Neural Networks and Engineered Features for Cardiac Arrhythmia Detection from ECG Recordings. Physiological Measurement 2019 [paper](https://www.ncbi.nlm.nih.gov/pubmed/30943458)

[4] Yuxi Zhou, Shenda Hong, Meng Wu, Junyuan Shang, Qingyun Wang, Junqing Xie, Hongyan Li. K-margin-based Residual-convolution-recurrent Neural Network for Atrial Fibrillation Detection. International Joint Conference on Artificial Intelligence (IJCAI) 2019 [paper](https://www.ijcai.org/proceedings/2019/0839.pdf)

[5] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, Ion Stoica: Ray: A Distributed Framework for Emerging AI Applications. OSDI 2018: 561-577