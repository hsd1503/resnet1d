

# Introduction

PyTorch implementations of several SOTA backbone deep neural networks (such as ResNet [1], ResNeXt [2], RegNet [3]) on one-dimensional (1D) signal/time-series data. 

If you use this code in your work, please cite our paper

```
@inproceedings{hong2020holmes,
  title={HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units},
  author={Hong, Shenda and Xu, Yanbo and Khare, Alind and Priambada, Satria and Maher, Kevin and Aljiffry, Alaa and Sun, Jimeng and Tumanov, Alexey},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1614--1624},
  year={2020}
}
```

# Usage

```
# test on synthetic data, no data download required
python test_synthetic.py

# test on PhysioNet/CinC Challenge 2017 data
# need prepare data first, or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
# Please see comment in code for details
python test_physionet.py

# training or serving model with Ray [4], need install Ray first: https://github.com/ray-project/ray
# Please see comment in code for details
python test_ray.py
```

model_detail/ shows model architectures

In ```trained_model```, we also provide a model ```model.pth``` trained using Challenge 2017 data. The model's parameters is as follows:
```
model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1.0,
        filter_list = [64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list = [2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        n_classes=4)
```

We can directly load this model by:
```
model = torch.load('model.pth')
```
The below image shows the change of training loss with epoch.
![image](https://user-images.githubusercontent.com/67882404/144691635-ea14e210-81f9-4889-9cbd-dd1a97290ca4.png)

Confusion Matrix on Validation Set:
| | Normal | AF | Others | Noisy |
| ---- | ---- | ---- | ---- | ---- |
|**Normal** | 0.878| 0.017 | 0.069 | 0.036 |
| **AF** | 0.032 | 0.861 | 0.077 | 0.030 |
| **Others** | 0.310 | 0.126 | 0.515 | 0.049 |
| **Noisy** | 0.107 | 0.058 | 0.020 | 0.815|

Other Metrics on Validation Set:
|AUC|F1|ACC|
|----|----|----|
|0.931|0.762|0.769|

# Requirements

Required: Python 3.7.5, PyTorch 1.2.0, torchsummary

Optional: Ray 0.8.0

# Application 1: ECG Classification (PhysioNet/CinC Challenge 2017)

The initial code has been used in our previous work [5,6,7] for deep feature extraction, which won one of the First place (F1=0.83) of this Challenge. The original tensorflow (tflearn) version can be found at https://github.com/hsd1503/ENCASE . 

Dataset: "AF Classification from a short single lead ECG recording". Data can be found at https://archive.physionet.org/challenge/2017/#challenge-data Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv , or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf .

This repository also contains data preprocessing code, please see util.py for details.

# Application 2: Health Monitoring in Intensive Care Units (KDD 20)

We built a set of models (called model zoo) for ensemble serving in Intensive Care Units. Please see more at [8]. 

# References


[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR 2016 [paper](https://arxiv.org/abs/1512.03385)

[2] Saining Xie, Ross B. Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017 [paper](https://arxiv.org/abs/1611.05431)

[3] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár. Designing network design spaces. CVPR 2020 [paper](https://arxiv.org/pdf/2003.13678.pdf)

[4] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, Ion Stoica: Ray: A Distributed Framework for Emerging AI Applications. OSDI 2018 [paper](https://www2.cs.uic.edu/~brents/cs494-cdcs/papers/ray.pdf)

[5] Shenda Hong, Meng Wu, Yuxi Zhou, Qingyun Wang, Junyuan Shang, Hongyan Li, Junqing Xie. ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks. Computing in Cardiology (CinC) Conference 2017 [paper](http://www.cinc.org/archives/2017/pdf/178-245.pdf)

[6] Shenda Hong, Yuxi Zhou, Meng Wu, Qingyun Wang, Junyuan Shang, Hongyan Li and Junqing Xie. Combining Deep Neural Networks and Engineered Features for Cardiac Arrhythmia Detection from ECG Recordings. Physiological Measurement 2019 [paper](https://www.ncbi.nlm.nih.gov/pubmed/30943458)

[7] Yuxi Zhou, Shenda Hong, Meng Wu, Junyuan Shang, Qingyun Wang, Junqing Xie, Hongyan Li. K-margin-based Residual-convolution-recurrent Neural Network for Atrial Fibrillation Detection. IJCAI 2019 [paper](https://www.ijcai.org/proceedings/2019/0839.pdf)

[8] Shenda Hong#, Yanbo Xu#, Alind Khare#, Satria Priambada#, Kevin Maher, Alaa Aljiffry, Jimeng Sun, Alexey Tumanov. HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units. KDD 2020 [paper](https://arxiv.org/pdf/2008.04063.pdf), [code](https://github.com/hsd1503/HOLMES)


