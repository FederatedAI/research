# Federated Learning Research at Webank AI

This repository contains research projects in federated learning from Webank AI group. It includes:

1. [Datasets](#datasets). Preprocessing codes of datasets we used and developed for federated learning research. 
2. [Publications](#publications). Implementation codes of our publications.
3. [Projects](#projects). Other projects in federated learning.

## NEWS

<b>2022-01-15 </b> [#Call for papers. https://www.computer.org/digital-library/journals/bd/call-for-papers-special-issue-on-trustable-verifiable-and-auditable-federated-learning
](https://www.computer.org/digital-library/journals/bd/call-for-papers-special-issue-on-trustable-verifiable-and-auditable-federated-learning)

<b>2021-11-12 </b> [#Call for papers. International Workshop on Trustable, Verifiable and Auditable Federated Learning in Conjunction with AAAI 2022 (FL-AAAI-22)
](http://federated-learning.org/fl-aaai-2022/)

<b>2021-10-01 (ET).</b> [#Call for papers. New Frontiers in Federated Learning:
Privacy, Fairness, Robustness, Personalization and Data Ownership Workshop
in Conjunction with NeurIPS 2021
](https://neurips2021workshopfl.github.io/NFFL-2021/index.html)

<b>2021-06-20.</b> [#Call for papers. International Workshop on Federated and Transfer Learning for Data Sparsity and Confidentiality
in Conjunction with IJCAI 2021 (FTL-IJCAI'21)
](http://federated-learning.org/fl-ijcai-2021/)

<b>2021-06-10.</b> [#Call for papers. International Workshop on Federated Learning for User Privacy and Data Confidentiality
in Conjunction with ICML 2021 (FL-ICML'21)
](http://federated-learning.org/fl-icml-2021/)

<b>2020-12-18.</b> [#Call for papers. ACM Transactions on Intelligent Systems and Technology, Special Issue on Federated Learning: Algorithms, Systems, and Applications
](https://dl.acm.org/pb-assets/static_journal_pages/tist/cfps/tist-si-cfp-12-2020-federated-learning2-1608108076107.pdf)

<b>2020-11-06.</b> The research directory is moved out from FATE project as an independent repository. Now you can get our latest research by staring this repo.


## Datasets
| Dataset | Description |
|-----------|------------------------|
|[Street Dataset](https://github.com/FederatedAI/research/tree/main/datasets/federated_object_detection_benchmark)|A real-world object detection dataset that annotates images captured by a set of street cameras based on object present in them, including 7 object categories.|
|[Fed_ModelNet40](https://github.com/FederatedAI/research/tree/main/datasets/Fed_Multiview_Gen)|It consists of images taken from various views of 3D models, and can be used for vertical federated learning research.|
|[NUS WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)|To simulate a vertical federated learning setting, the image features of samples is put on one party and the textual tags on another party.|
|[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)|CheXpert is a large dataset of chest X-rays and can be used for vertical federated learning research.|


## Publications
Our publications are categorized  as below:

- [<b>Highlight</b>](#highlight). Our newly and highly interesting works.
- [<b>Paradigm</b>](#paradigm). Various types of federated learning schemes. 
- [<b>Security</b>](#Privacy). Data privacy and model security.
- [<b>Efficiency</b>](#Efficiency). Communication and computation efficiency, data distribution heterogeneity, system interpretability and Incentive Mechanism.  
- [<b>Incentive</b>](#Incentive). Incentive Mechanism.  
- [<b>Application</b>](#application). Federated learning in real-world applications.
- [<b>Dataset</b>](#dataset). Datasets for federated learning research.
- [<b>Survey</b>](#survey). Survey on various topics of federated learning.

-------------------

### Highlight
| Title| Code| Description|
|-----|-----|-----|
|[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)|[code](https://github.com/FederatedAI/FATE/blob/master/python/federatedml/ensemble)|IEEE intelligent Systems 2021, widely-used federated tree-boosting algorithm|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)|[code](https://github.com/FederatedAI/FATE/tree/develop-1.5/python/federatedml/transfer_learning/hetero_ftl)|IEEE intelligent Systems 2020, the first federated transfer learning paper|
|[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202)|[code](https://github.com/FederatedAI/FedVision)|AAAI 2020, Innovative Application of Artificial Intelligence Award from AAAI in 2020|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)| |ACM TIST 2019, the 3rd most cited federated learning paper|


### Paradigm
| Title| Code| Description|
|-----|-----|-----|
|[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](https://arxiv.org/abs/2005.11418)| |ICML FL workshop 2020|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)|[code](https://github.com/FederatedAI/FATE/tree/develop-1.5/python/federatedml/transfer_learning/hetero_ftl)|IEEE intelligent Systems 2020|
|[FedMVT: Semi-supervised Vertical Federated Learning with MultiView Training](https://arxiv.org/abs/2008.10838) | |IJCAI FL workshop 2020|
|[Federated Transfer Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1910.06001)|[code](https://github.com/FederatedAI/research/tree/main/publications/FTRL)| |
|[Privacy-preserving Heterogeneous Federated Transfer Learning](https://ieeexplore.ieee.org/document/9005992)| |IEEE BigData 2019|
|[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)|[code](https://github.com/FederatedAI/FATE/blob/master/python/federatedml/ensemble)|IEEE intelligent Systems 2021|
|[Multi-Component Transfer Metric Learning for handling unrelated source domain samples](https://www.sciencedirect.com/science/article/abs/pii/S0950705120303877)| |Knowledge-Based Systems|
|[Cross-silo Federated Neural Architecture Search forHeterogeneous and Cooperative Systems](https://arxiv.org/pdf/2101.11896.pdf)|[code](https://github.com/FederatedAI/research/tree/main/publications/ss_vfnas)|Self-supervised Vertical Federated NAS|

### Security and Privacy
| Title| Code| Description|
|-----|-----|-----|
|[Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack](https://arxiv.org/abs/2102.04362)| |CVPR 2021|
|[Backdoor attacks and defenses in feature-partitioned collaborative learning](https://arxiv.org/abs/2007.03608)|[code](https://github.com/FederatedAI/research/tree/main/publications/vfl_backdoor)|ICML 2020 FL workshop|
|[Secure Federated Matrix Factorization](https://arxiv.org/abs/1906.05108)| |IEEE Intelligent Systems 2020|
|[Privacy-Preserving Deep Learning with SPDZ](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/papers/paper_3.pdf)| |The AAAI Workshop on PPAI|
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)| |USENIX 2020 ATC|
|[Privacy Threats Against Federated Matrix Factorization](https://arxiv.org/abs/2007.01587)| |IJCAI 2020 FL workshop|
|[Dynamic backdoor attacks against federated learning](https://arxiv.org/abs/2011.07429)| | |
|[Rethinking Privacy Preserving Deep Learning: How to Evaluate and Thwart Privacy Attacks](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_3)| |Springer Book 2020|
|[Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attack](https://proceedings.neurips.cc/paper/2019/hash/75455e062929d32a333868084286bb68-Abstract.html)|[code](https://github.com/kamwoh/DeepIPR)|NIPS 2019|
|[Abnormal client behavior detection in federated learning](https://arxiv.org/abs/1910.09933)| |NIPS workshop 2019|

### Efficiency
| Title| Code| Description|
|-----|-----|-----|
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)|[code](https://github.com/marcoszh/BatchCrypt)|2020 USENIX ATC 2020|
|[A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187)| |NeurIPS 2019 FL workshop|
|[RPN: A Residual Pooling Network for Efficient Federated Learning](https://arxiv.org/abs/2001.08600)| |ECAI 2020|
|[Secure and Efficient Federated Transfer Learning](https://arxiv.org/abs/1910.13271)| |IEEE BigData 2019|

### Incentive
| Title| Code| Description|
|-----|-----|-----|
|[A Fairness-aware Incentive Scheme for Federated Learning](https://dl.acm.org/doi/abs/10.1145/3375627.3375840)| |AIES 2020|
|[A Sustainable Incentive Scheme for Federated Learning](https://ieeexplore.ieee.org/document/9069185)| |IEEE Intelligent Systems |
|[A multi-player game for studying federated learning incentive schemes](https://www.ijcai.org/Proceedings/2020/769)| |IJCAI 2020|


### Application
| Title| Code| Description|
|-----|-----|-----|
|[Variation-Aware Federated Learning with Multi-Source Decentralized Medical Image Data](https://ieeexplore.ieee.org/document/9268161)| |IEEE Journal of Biomedical and Health Informatics 2020|
|[Fedml: A research library and benchmark for federated machine learning](https://arxiv.org/abs/2007.13518)|[code](https://github.com/FedML-AI/FedML)|NeurIPS 2020 FL workshop|
|[Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/abs/2004.12321)|[code](https://github.com/DashanGao/Federated-Transfer-Learning-for-EEG)|IEEE EMBC 2020|
|[Multi-Agent Visualization for Explaining Federated Learning](https://www.ijcai.org/Proceedings/2019/960)| |IJCAI 2019|
|[HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/abs/1909.05784)| |IJCAI FL workshop 2020|
|[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202)|[code](https://github.com/FederatedAI/FedVision)|AAAI 2020|
|[Fair and Explainable Dynamic Engagement of Crowd Workers](https://www.ijcai.org/Proceedings/2019/961)| |IJCAI 2019|
|[Privacy-Preserving Technology to Help Millions of People: Federated Prediction Model for Stroke Prevention](https://arxiv.org/abs/2006.10517)| |IJCAI 2020 FL workshop|

### Dataset
| Title| Code| Description|
|-----|-----|-----|
|[Real-World Image Datasets for Federated Learning](https://arxiv.org/abs/1910.11089)| [code](https://github.com/FederatedAI/research/tree/main/datasets/federated_object_detection_benchmark)|NIPS FL workshop 2019|


### Survey
| Title| Code| Description|
|-----|-----|-----|
|[Towards Personalized Federated Learning](https://arxiv.org/abs/2103.00710)| |IJCAI FL workshop 2021|
|[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)| |Foundations and Trends in Machine Learning 2021|
|[Threats to Federated Learning: A Survey](https://arxiv.org/abs/2003.02133)| |IJCAI FL workshop 2020|
|[Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective](https://arxiv.org/abs/2002.11545)| |ArXiv 2020|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)| |ACM TIST 2019|



## Projects
Currently, we are actively contributing to two projects, [FedML](https://github.com/FedML-AI/FedML) (research-origented) and [FATE](https://github.com/FederatedAI/FATE) (application-oriented).

### [<b>FedML</b>](https://github.com/FedML-AI/FedML)

FedML (Federated Machine Learning) is a research-oriented Federated Learning Library. It provides a plenty of out-of-the-box modules in federated learning, which greatly facilitates the development of new federated learning algorithms for researchers. We are co-contributor to this project and mainly maintain the part of vertical federated learning. 
- [Vertical federated learning](https://github.com/FedML-AI/FedML/tree/master/fedml_experiments/standalone/classical_vertical_fl)

  

### [<b>FATE</b>](https://github.com/FederatedAI/FATE)

FATE (Federated AI Technology Enabler) is an industrial grade Federated Learning framework. It has already incorporated many of our proposed methods and algorithms to enhance its security and efficiency under various federated learning scenarios. Some of the implemented algorithms are listed below:

- [SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)
- [A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)
- [A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187)
- [Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)


## License
[Apache 2.0 license](LICENSE).

