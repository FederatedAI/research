# Federated Learning Research at Webank AI

This repository contains research projects in federated learning from Webank AI group. It includes:

1. [Datasets](#datasets). Preprocessing codes of datasets we used and developed for federated learning research. 
2. [Publications](#publications). Implementation codes of our publications.
3. [Projects](#projects). Other projects in federated learning.

## NEWS
<b>2020-11-06.</b> The research directory is moved out from FATE project as an independent repository. Now you can get our latest research by staring this repo.


## Datasets
| Dataset | Description |
|-----------|------------------------|
|[Street Dataset](https://github.com/FederatedAI/research/tree/main/datasets/federated_object_detection_benchmark)|A real-world object detection dataset that annotates images captured by a set of street cameras based on object present in them, including 7 object categories.|
|[Fed_ModelNet40](https://github.com/FederatedAI/research/tree/main/datasets/Fed_Multiview_Gen)|It consists of images taken from various views of 3D models, and can be used for vertical federated learning research.|
|NUS WIDE|To simulate a vertical federated learning setting, the image features of samples is put on one party and the textual tags on another party.|
|[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)|CheXpert is a large dataset of chest X-rays and can be used for vertical federated learning research.|


## Publications
Our publications are categorized  as below:

[<b>Highlight</b>](#highlight). Our newly and highly interesting works.

[<b>Paradigm</b>](#paradigm). Various types of federated learning schemes. 

[<b>Security</b>](#security). Privacy and data security.

[<b>System performance</b>](#system-performance). Communication and computation efficiency, data distribution heterogeneity, system interpretability and Incentive Mechanism.  

[<b>Application</b>](#application). Federated learning in real-world applications.

[<b>Dataset</b>](#dataset). Datasets for federated learning research.

[<b>Survey</b>](#survey).

### Highlight
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)||ACM TIST|
|[Backdoor attacks and defenses in feature-partitioned collaborative learning](https://arxiv.org/abs/2007.03608)|[code](https://github.com/FederatedAI/research/tree/main/publications/vfl_backdoor)|ICML 2020 FL workshop|


### Paradigm
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Secure and Efficient Federated Transfer Learning](https://arxiv.org/abs/1910.13271)||IEEE BigData 2019|
|[Privacy-preserving heterogeneous federated transfer learning](https://ieeexplore.ieee.org/document/9005992)||IEEE BigData 2019|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)||IEEE intelligent Systems|
|[FedMVT: Semi-supervised Vertical Federated Learning with MultiView Training](https://arxiv.org/abs/2008.10838) ||IJCAI 2020 FL workshop|
|[Federated Transfer Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1910.06001)|[code](https://github.com/FederatedAI/research/tree/main/publications/FTRL)||


### Security
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Backdoor attacks and defenses in feature-partitioned collaborative learning](https://arxiv.org/abs/2007.03608)|[code](https://github.com/FederatedAI/research/tree/main/publications/vfl_backdoor)|ICML 2020 FL workshop|
|[Privacy-Preserving Deep Learning with SPDZ](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/papers/paper_3.pdf)||The AAAI Workshop on PPAI|
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)||USENIX 2020 ATC|
|[Privacy Threats Against Federated Matrix Factorization](https://arxiv.org/abs/2007.01587)||IJCAI 2020 FL workshop|


### System performance
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[A Fairness-aware Incentive Scheme for Federated Learning](https://dl.acm.org/doi/abs/10.1145/3375627.3375840)||AIES 2020|
|[A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187) ||NeurIPS 2019 FL workshop|
|[A Sustainable Incentive Scheme for Federated Learning](https://ieeexplore.ieee.org/document/9069185)||IEEE Intelligent Systems |
|[Multi-Component Transfer Metric Learning for handling unrelated source domain samples](https://www.sciencedirect.com/science/article/abs/pii/S0950705120303877)||Knowledge-Based Systems|
|[A multi-player game for studying federated learning incentive schemes](https://www.ijcai.org/Proceedings/2020/769)||IJCAI 2020|
|[Secureboost: A lossless federated learning framework](https://arxiv.org/abs/1901.08755)|||
|[RPN: A Residual Pooling Network for Efficient Federated Learning](https://arxiv.org/abs/2001.08600)|||
|[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](https://arxiv.org/abs/2005.11418)|||

### Application
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Fedml: A research library and benchmark for federated machine learning](https://arxiv.org/abs/2007.13518)|[code](https://github.com/FedML-AI/FedML)|NeurIPS 2020 FL workshop|
|[Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/abs/2004.12321)|[code](https://github.com/DashanGao/Federated-Transfer-Learning-for-EEG)|IEEE EMBC 2020|
|[Multi-Agent Visualization for Explaining Federated Learning](https://www.ijcai.org/Proceedings/2019/960)||IJCAI 2019|
|[HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/abs/1909.05784)||IJCAI 2020 FL workshop|
|[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202)||AAAI 2020|
|[Fair and Explainable Dynamic Engagement of Crowd Workers](https://www.ijcai.org/Proceedings/2019/961)||IJCAI 2019|
|[Privacy-Preserving Technology to Help Millions of People: Federated Prediction Model for Stroke Prevention](https://arxiv.org/abs/2006.10517)||IJCAI 2020 FL workshop|
|[Abnormal client behavior detection in federated learning](https://arxiv.org/abs/1910.09933)|||


### Dataset
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Real-World Image Datasets for Federated Learning](https://arxiv.org/abs/1910.11089)| [code](https://github.com/FederatedAI/research/tree/main/datasets/federated_object_detection_benchmark)||


### Survey
| Paper| Code| Description|
|----------------------------------------------------|-----|-----|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)||ACM TIST|
|[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)|||



## Projects


## License
[Apache 2.0 license](LICENSE).
