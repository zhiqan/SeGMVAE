# SeGMVAE
## [A Semi-supervised Gaussian Mixture Variational Autoencoder method for few-shot fine-grained fault diagnosis](https://doi.org/10.1016/j.neunet.2024.106482)
In practical engineering, obtaining labeled high-quality fault samples poses challenges. Conventional fault diagnosis methods based on deep learning struggle to discern the underlying causes of mechanical faults from a fine-grained perspective, due to the scarcity of annotated data. To tackle those issue, we propose a novel semi-supervised Gaussian Mixed Variational Autoencoder method, SeGMVAE, aimed at acquiring unsupervised representations that can be transferred across fine-grained fault diagnostic tasks, enabling the identification of previously unseen faults using only the small number of labeled samples. Initially, Gaussian mixtures are introduced as a multimodal prior distribution for the Variational Autoencoder. This distribution is dynamically optimized for each task through an expectation–maximization (EM) algorithm, constructing a latent representation of the bridging task and unlabeled samples. Subsequently, a set variational posterior approach is presented to encode each task sample into the latent space, facilitating meta-learning. Finally, semi-supervised EM integrates the posterior of labeled data by acquiring task-specific parameters for diagnosing unseen faults. Results from two experiments demonstrate that SeGMVAE excels in identifying new fine-grained faults and exhibits outstanding performance in cross-domain fault diagnosis across different machines. 
# The main idea of the method in this paper comes from [Meta-GMVAE](https://openreview.net/forum?id=wS0UFjsNYjn).
# CWRU fine-grained fault dataset
## Baidu Netdisk link：https://pan.baidu.com/s/17iIKPcur3oNeTx8-NhRhLQ 
## 提取码：523s


# If it is helpful for your research, please kindly cite this work:
﻿
```html

@inproceedings{lee2020meta,
  title={Meta-gmvae: Mixture of gaussian vae for unsupervised meta-learning},
  author={Lee, Dong Bok and Min, Dongchan and Lee, Seanie and Hwang, Sung Ju},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@article{ZHAO2024106482,
title = {A Semi-supervised Gaussian Mixture Variational Autoencoder method for few-shot fine-grained fault diagnosis},
journal = {Neural Networks},
pages = {106482},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106482},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024004064},
author = {Zhiqian Zhao and Yeyin Xu and Jiabin Zhang and Runchao Zhao and Zhaobo Chen and Yinghou Jiao}
}
```
