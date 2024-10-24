# Trustworthy Federated Learning Research

This repository contains research works and projects on trustworthy federated learning. It includes:

1. [Datasets](#datasets). Preprocessing codes of datasets we used and developed for federated learning research. 
2. [Publications](#publications). Implementation codes of our publications.
3. [Projects](#projects). Other projects in federated learning.

## Federated Learning Portal

This [Federated Learning Portal](http://federated-learning.org/) keeps track of books, workshops, conference special tracks, journal special issues, standardization effort and other notable events related to the field of Federated Learning (FL).


## Datasets
| Dataset | Description |
|-----------|------------------------|
|[Street Dataset](https://github.com/FederatedAI/research/tree/main/datasets/federated_object_detection_benchmark)|A real-world object detection dataset that annotates images captured by a set of street cameras based on object present in them, including 7 object categories.|
|[Fed_ModelNet40](https://github.com/FederatedAI/research/tree/main/datasets/Fed_Multiview_Gen)|It consists of images taken from various views of 3D models, and can be used for vertical federated learning research.|
|[NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)|To simulate a vertical federated learning setting, the image features of samples is put on one party and the textual tags on another party.|
|[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)|CheXpert is a large dataset of chest X-rays and can be used for vertical federated learning research.|


## Publications
Our publications are categorized  as below:

- [<b>Highlight</b>](#highlight). Papers that have high impact or we recommend to read.
- [<b>FTL-FM</b>](#FTL-FM). Grounding foundation models via federated transfer learning.
- [<b>Security and Privacy</b>](#Security). Security and privacy attacks and defenses.
- [<b>Intellectual Property Protection</b>](#Intellectual). Intellectual property protection and ownership verification (on model or data).
- [<b>Effectiveness</b>](#Effectiveness). Various algorithms aim to improve the effectiveness of FL. 
- [<b>Efficiency</b>](#Efficiency). Communication and computation efficiency.  
- [<b>Incentive</b>](#Incentive). Incentive Mechanism.
- [<b>Theory</b>](#Theory). Theoretical work of federated learning.  
- [<b>Application</b>](#application). Federated learning in real-world applications.
- [<b>Dataset</b>](#dataset). Datasets for federated learning research.
- [<b>Survey</b>](#survey). Survey on various topics of federated learning.

-------------------

### High Citation Papers
| Title| Code| Description| Semantic Scholar Citation | Google Scholar Citation (by 01/10/2023)|
|-----|-----|-----|-----|-----|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)| |ACM TIST 2019, the 3rd most cited federated learning paper|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/c38e83a9d0fe7005183f3c5bedf32a23df1b579b?fields=citationCount)](https://www.semanticscholar.org/paper/Federated-Machine-Learning%3A-Concept-and-Yang-Liu/c38e83a9d0fe7005183f3c5bedf32a23df1b579b)| 2995	| 
|[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)| |Foundations and Trends in Machine Learning 2021|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/07912741c6c96e6ad5b2c2d6c6c3b2de5c8a271b?fields=citationCount)](https://www.semanticscholar.org/paper/Advances-and-Open-Problems-in-Federated-Learning-Kairouz-McMahan/07912741c6c96e6ad5b2c2d6c6c3b2de5c8a271b)| 2711	| 
|[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)|[code](https://github.com/FederatedAI/FATE/blob/master/python/federatedml/ensemble)|IEEE intelligent Systems 2021, widely-used federated tree-boosting algorithm, best paper award|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/f71687796caa7d2a5d68406b53b51cb9ed57f586?fields=citationCount)](https://www.semanticscholar.org/paper/SecureBoost%3A-A-Lossless-Federated-Learning-Cheng-Fan/f71687796caa7d2a5d68406b53b51cb9ed57f586)|333	|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)|[code](https://github.com/FederatedAI/FATE/tree/develop-1.5/python/federatedml/transfer_learning/hetero_ftl)|IEEE Intelligent Systems 2020, the first federated transfer learning paper|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/cc8379abcda8faa78e2c5e17deb96785ea447461?fields=citationCount)](https://www.semanticscholar.org/paper/A-Secure-Federated-Transfer-Learning-Framework-Liu-Kang/cc8379abcda8faa78e2c5e17deb96785ea447461)  |338	|
|[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202)|[code](https://github.com/FederatedAI/FedVision)|AAAI 2020, Innovative Application of Artificial Intelligence Award from AAAI in 2020|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/1de7dae4b03fb3ab1eeffc45d2fd761538962d5c?fields=citationCount)](https://www.semanticscholar.org/paper/FedVision%3A-An-Online-Visual-Object-Detection-by-Liu-Huang/1de7dae4b03fb3ab1eeffc45d2fd761538962d5c) |144	| 
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)|[code](https://github.com/marcoszh/BatchCrypt)|2020 USENIX ATC 2020|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/c783cdc03a32e5094affa7eef710459aac599aaf?fields=citationCount)](https://www.semanticscholar.org/paper/BatchCrypt%3A-Efficient-Homomorphic-Encryption-for-Zhang-Li/c783cdc03a32e5094affa7eef710459aac599aaf)|261	|
|[A Fairness-aware Incentive Scheme for Federated Learning](https://dl.acm.org/doi/abs/10.1145/3375627.3375840)| |AIES 2020|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/e64e685190357333513aa3bbaa7e446c7683bfbe?fields=citationCount)](https://www.semanticscholar.org/paper/A-Fairness-aware-Incentive-Scheme-for-Federated-Yu-Liu/e64e685190357333513aa3bbaa7e446c7683bfbe)|117	|
|[Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attack](https://proceedings.neurips.cc/paper/2019/hash/75455e062929d32a333868084286bb68-Abstract.html)|[code](https://github.com/kamwoh/DeepIPR)|NIPS 2019|[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/5d3fd7275e5ce39d7ddae12894f5ca1377db6b01?fields=citationCount)](https://www.semanticscholar.org/paper/Rethinking-Deep-Neural-Network-Ownership-Embedding-Fan-Ng/5d3fd7275e5ce39d7ddae12894f5ca1377db6b01)|118| 
|[Towards Personalized Federated Learning](https://arxiv.org/abs/2103.00710)| |IEEE Transactions on Neural Networks and Learning Systems 2022 |[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org/graph/v1/paper/481dd25896ac531707870c9b8c179cce20013401?fields=citationCount)](https://www.semanticscholar.org/paper/Towards-Personalized-Federated-Learning-Tan-Yu/481dd25896ac531707870c9b8c179cce20013401)|115| 

### Highlight Paper
| Title| Code| Description| 
|-----|-----|-----|
|[Grounding Foundation Models through Federated Transfer Learning: A General Framework](https://arxiv.org/abs/2311.17431)||Preprint|
|[Optimizing Privacy, Utility and Efficiency in Constrained Multi-Objective Federated Learning](https://arxiv.org/abs/2305.00312)||ACM TIST 2024|
|[Probably Approximately Correct Federated Learning](https://arxiv.org/abs/2304.04641)||Preprint|
|[Trading Off Privacy, Utility and Efficiency in Federated Learning](https://arxiv.org/abs/2209.00230)||ACM TIST 2023|
|[No Free Lunch Theorem for Security and Utility in Federated Learning](https://arxiv.org/pdf/2203.05816.pdf)||ACM TIST 2022|
|[FedIPR: Ownership Verification for Federated Deep Neural Network Models](https://arxiv.org/pdf/2109.13236.pdf)|[code](https://github.com/FederatedAI/research/tree/main/publications/FedIPR)|IEEE Transactions on Pattern Analysis and Machine Intelligence 2022|
|[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)|[code](https://github.com/FederatedAI/FATE/blob/master/python/federatedml/ensemble)|IEEE intelligent Systems 2021, widely-used federated tree-boosting algorithm|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)|[code](https://github.com/FederatedAI/FATE/tree/develop-1.5/python/federatedml/transfer_learning/hetero_ftl)|IEEE intelligent Systems 2020, the first federated transfer learning paper|
|[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202)|[code](https://github.com/FederatedAI/FedVision)|AAAI 2020, Innovative Application of Artificial Intelligence Award from AAAI in 2020|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)| |ACM TIST 2019, the 3rd most cited federated learning paper|

### FTL-FM
| Title| Code| Description|
|-----|-----|-----|
|[Grounding Foundation Models through Federated Transfer Learning: A General Framework](https://arxiv.org/abs/2311.17431)||Preprint|
|[FATE-LLM: A Industrial Grade Federated Learning Framework for Large Language Models](https://arxiv.org/abs/2310.10049)||LLM@IJCAI'23|


### Security and Privacy
| Title| Code| Description|
|-----|-----|-----|
|[Achieving Provable Byzantine Fault-Tolerance in a Semi-honest Federated Learning Setting](https://dl.acm.org/doi/abs/10.1007/978-3-031-33377-4_32) |  | PAKDD 2023 |
|[FedPass: Privacy-Preserving Vertical Federated Deep Learning with Adaptive Obfuscation](https://arxiv.org/pdf/2301.12623.pdf) | | IJCAI 2023 |
|[A Framework for Evaluating Privacy-Utility Trade-off in Vertical Federated Learning](https://arxiv.org/abs/2209.03885) | | preprint |
|[FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning](https://arxiv.org/abs/2111.08211)|[code](https://github.com/FederatedAI/research/tree/main/publications/FedCG)|IJCAI 2022|
|[Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning](https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe)|[code](https://github.com/FLAIR-THU/TrustworthyFederatedLearning)|IEEE Transactions on Big Data|
|[Secure Federated Matrix Factorization](https://arxiv.org/abs/1906.05108)| |IEEE Intelligent Systems 2020|
|[Privacy-Preserving Deep Learning with SPDZ](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/papers/paper_3.pdf)| |The AAAI Workshop on PPAI|
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)| |USENIX 2020 ATC|
|[Privacy Threats Against Federated Matrix Factorization](https://arxiv.org/abs/2007.01587)| |IJCAI 2020 FL workshop|
|[Dynamic backdoor attacks against federated learning](https://arxiv.org/abs/2011.07429)| | |
|[Rethinking Privacy Preserving Deep Learning: How to Evaluate and Thwart Privacy Attacks](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_3)| |Springer Book 2020|
|[Abnormal client behavior detection in federated learning](https://arxiv.org/abs/1910.09933)| |NIPS workshop 2019|

### Intellectual Property Protection 
| Title| Code| Description|
|-----|-----|-----|
|[FedTracker: Furnishing Ownership Verification and Traceability for Federated Learning Model](https://arxiv.org/pdf/2211.07160.pdf)|| IEEE Transactions on Dependable and Secure Computing 2024|
|[FedIPR: Ownership Verification for Federated Deep Neural Network Models](https://arxiv.org/pdf/2109.13236.pdf)||IEEE TPAMI, 2022|
|[Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack](https://arxiv.org/abs/2102.04362)| |CVPR 2021|
|[Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attack](https://proceedings.neurips.cc/paper/2019/hash/75455e062929d32a333868084286bb68-Abstract.html)|[code](https://github.com/kamwoh/DeepIPR)|NIPS 2019|

### Effectiveness
| Title| Code| Description|
|-----|-----|-----|
|[FedHSSL: A Hybrid Self-Supervised Learning Framework for Vertical Federated Learning](https://arxiv.org/abs/2208.08934) | [code](https://github.com/jorghyq2016/FedHSSL) |IEEE Transactions on Big Data 2024|
|[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](https://arxiv.org/abs/2005.11418)| |ICML FL workshop 2020|
|[A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)|[code](https://github.com/FederatedAI/FATE/tree/develop-1.5/python/federatedml/transfer_learning/hetero_ftl)|IEEE intelligent Systems 2020|
|[FedCVT: Semi-supervised Vertical Federated Learning with Cross-View Training](https://arxiv.org/abs/2008.10838) | |ACM TIST 2022|
|[Federated Transfer Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1910.06001)|[code](https://github.com/FederatedAI/research/tree/main/publications/FTRL)| Federated and Transfer Learning Book |
|[Privacy-preserving Heterogeneous Federated Transfer Learning](https://ieeexplore.ieee.org/document/9005992)| |IEEE BigData 2019|
|[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)|[code](https://github.com/FederatedAI/FATE/blob/master/python/federatedml/ensemble)|IEEE intelligent Systems 2021|
|[Multi-Component Transfer Metric Learning for handling unrelated source domain samples](https://www.sciencedirect.com/science/article/abs/pii/S0950705120303877)| |Knowledge-Based Systems|
|[Cross-silo Federated Neural Architecture Search for Heterogeneous and Cooperative Systems](https://arxiv.org/pdf/2101.11896.pdf)|[code](https://github.com/FederatedAI/research/tree/main/publications/ss_vfnas)|Federated and Transfer Learning Book|

### Efficiency
| Title| Code| Description|
|-----|-----|-----|
|[FLASHE: Additively Symmetric Homomorphic Encryption for Cross-Silo Federated Learning](https://arxiv.org/abs/2109.00675)| |Arxiv 2021|
|[Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)|[code](https://github.com/marcoszh/BatchCrypt)|2020 USENIX ATC 2020|
|[FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features](https://ieeexplore.ieee.org/document/9855231)|[code](https://github.com/yankang18/FedBCD)|IEEE Transactions on Signal Processing 2022|
|[RPN: A Residual Pooling Network for Efficient Federated Learning](https://arxiv.org/abs/2001.08600)| |ECAI 2020|
|[Secure and Efficient Federated Transfer Learning](https://arxiv.org/abs/1910.13271)| |IEEE BigData 2019|

### Incentive
| Title| Code| Description|
|-----|-----|-----|
|Contribution-Aware Federated Learning for Smart Healthcare| |IAAI 2022|
|[A Fairness-aware Incentive Scheme for Federated Learning](https://dl.acm.org/doi/abs/10.1145/3375627.3375840)| |AIES 2020|
|[A Sustainable Incentive Scheme for Federated Learning](https://ieeexplore.ieee.org/document/9069185)| |IEEE Intelligent Systems |
|[A multi-player game for studying federated learning incentive schemes](https://www.ijcai.org/Proceedings/2020/769)| |IJCAI 2020|

### Theory
| Title| Code| Description| 
|-----|-----|-----|
|[Probably Approximately Correct Federated Learning](https://arxiv.org/abs/2304.04641)||Preprint|
|[A Game-theoretic Framework for Federated Learning](https://arxiv.org/abs/2304.05836)||ACM TIST 2024|
|[Trading Off Privacy, Utility and Efficiency in Federated Learning](https://arxiv.org/abs/2209.00230)||ACM TIST 2023|
|[No Free Lunch Theorem for Security and Utility in Federated Learning](https://arxiv.org/pdf/2203.05816.pdf)||ACM TIST 2022|

### Application
| Title| Code| Description|
|-----|-----|-----|
|[Amalur: Data Integration Meets Machine Learning](https://arxiv.org/abs/2205.09681)| |ICDE 2023 (Vision paper)|
|[StarFL: Hybrid Federated Learning Architecture for Smart Urban Computing](https://dl.acm.org/doi/abs/10.1145/3467956)| |ACM TIST 2021|
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
|[Vertical Federated Learning: Concepts, Advances, and Challenges](https://arxiv.org/abs/2211.12814)| |TKDE 2024|
|[A Survey on Heterogeneous Federated Learning](https://arxiv.org/abs/2210.04505)| |Preprint|
|[Toward Responsible AI: An Overview of Federated Learning for User-centered Privacy-preserving Computing](https://dl.acm.org/doi/abs/10.1145/3485875)| |ACM TIST 2022|
|[Towards Personalized Federated Learning](https://arxiv.org/abs/2103.00710)| |IEEE Transactions on Neural Networks and Learning Systems|
|[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)| |Foundations and Trends in Machine Learning 2021|
|[Threats to Federated Learning: A Survey](https://arxiv.org/abs/2003.02133)| |IJCAI FL workshop 2020|
|[Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective](https://arxiv.org/abs/2002.11545)| |ArXiv 2020|
|[Federated machine learning: Concept and applications](https://arxiv.org/abs/1902.04885)| |ACM TIST 2019|



## Projects
Currently, we are actively contributing to two projects, [FedML](https://github.com/FedML-AI/FedML) (research-origented) and [FATE](https://github.com/FederatedAI/FATE) (application-oriented).


### [<b>FATE</b>](https://github.com/FederatedAI/FATE)

FATE (Federated AI Technology Enabler) is an industrial grade Federated Learning framework. It has already incorporated many of our proposed methods and algorithms to enhance its security and efficiency under various federated learning scenarios. Some of the implemented algorithms are listed below:

- [SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755)
- [A Secure Federated Transfer Learning Framework](https://ieeexplore.ieee.org/document/9076003)
- [A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187)
- [Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)


## License
[Apache 2.0 license](LICENSE).

