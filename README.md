# This is a paper reading log.

Log my master learning period.



before 2024/6/14

**title:** FedDWA: Personalized Federated Learning with Dynamic Weight Adjustment

**source:** [[2305.06124\] FedDWA: Personalized Federated Learning with Dynamic Weight Adjustment (arxiv.org)](https://arxiv.org/abs/2305.06124)

主要贡献就是在PFL中，简化了两个客户端的距离公式，使用拉格朗日数乘法求解最优解。使用了多做一次训练作为guidance。



---



2024/6/14

**title:** Selective knowledge sharing for privacy preserving federated distillation without a good teacher  

**DOI:** https://doi.org/10.1038/s41467-023-44383-9  

nature communications

code: https://github.com/shaojiawei07/ Selective-FD  

![image-20240614230332731](img/image-20240614230332731.png)



***

2024/6/16

**title**: Federated Linear Contextual Bandits with Heterogeneous Clients  

**code**: https://github.com/blaserethan/HetoFedBandit

不是很懂



---

2024/6/17

**title**: Towards Data-Independent Knowledge Transfer in Model-Heterogeneous Federated Learning 

**source**: IEEE TRANSACTIONS ON COMPUTERS, VOL. 72, NO. 10, OCTOBER 2023  

GAN+FD

![image-20240617173920104](img/image-20240617173920104.png)



---



2024/6/18

**title**: Scheduling Algorithms for Federated Learning With Minimal Energy Consumption  

**source**: IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS, VOL. 34, NO. 4, APRIL 2023  

method: (MC)2MKP and dynamic programming

this paper focuses on the energy cost in FL

跟联邦学习关系不大，没有考虑Non-IID，而且没有FL训练效果 

不推荐看



**title**: PrivAim: A Dual-Privacy Preserving and Quality-Aware Incentive Mechanism for Federated Learning  

**source**: IEEE TRANSACTIONS ON COMPUTERS, VOL. 72, NO. 7, JULY 2023  

对隐私感兴趣可以看，对具体的FL训练没有实质性进展



**title**: FedRFQ: Prototype-Based Federated Learning With Reduced Redundancy, Minimal Failure, and Enhanced Quality  

**source**: IEEE TRANSACTIONS ON COMPUTERS, VOL. 73, NO. 4, APRIL 2024  

conclude: 原型+softpool+practical byzantine fault-tolerance

主要突破就是减少了通信量 acc提升量还行



在GitHub上的浏览位置为

![image-20240618235637521](img/image-20240618235637521.png)

---



下载了相对比较高质量的文章耗时半天。

---



2024/6/19

**title**: User-Distribution-Aware Federated Learning for Efficient Communication and Fast Inference

**source**: IEEE TRANSACTIONS ON COMPUTERS, VOL. 73, NO. 4, APRIL 2024

个人见解：FL框架没什么创新。只是考虑了一个edge或者cluster的重要性，以及设计了LB-Net来减少通讯量。没啥贡献。亮点可能就是证明吧。

可以参考证明思路



---

2024/6/20



**title:** Coalitional FL: Coalition Formation and Selection in Federated Learning with Heterogeneous Data

**source**: DOI 10.1109/TMC.2024.3375325

好文章。分析了异构性和model acc的关系。



---

2024/6/23

**title**: A Coalition Formation Game Approach for Personalized Federated Learning

**source**: arXiv:2202.02502v2

Good Paper

recommended



**title**: A Data Privacy Protection Scheme Integrating  Federal Learning and Secret Sharing

**source**: 2023 IEEE 5th International Conference on Power, Intelligent Computing and Systems (ICPICS) DOI: 10.1109/ICPICS58376.2023.10235406

对数据使用密码学加密，学过密码学就知道这个特别基础 

本文没有什么贡献。



**title**:  FedMut: Generalized Federated Learning via Stochastic Mutation

**source**: TheThirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)

对参数进行变异后进行聚合。既然对参数进行变异，其实可以启发使用遗传算法。



**title**:  Competitive-Cooperative Multi-Agent Reinforcement Learning for Auction-based Federated Learning

**source**: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)

介绍了AFL。可以参考其定义

个人绝对就是调整那个τ超参数来管理整个RL

---



2024/6/24

**title**: FEDGKD: Toward Heterogeneous Federated Learning via Global Knowledge Distillation

**source**: IEEE TRANSACTIONS ON COMPUTERS, VOL. 73, NO. 1, JANUARY 2024

特点就是global模型将历史模型做了一个buffer，进行平均聚合



**title**: FairTrade: Achieving Pareto-Optimal Trade-Offs between Balanced Accuracy and Fairness in Federated Learning

**source**: TheThirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)

多目标优化的FL



---

2024/7/12

**title:**Boosting with Multiple Sources

**source**: 35th Conference on Neural Information Processing Systems (NeurIPS 2021).

集成学习+fed 里面的Q函数可以研究研究



**title**:  ADAPTIVE FEDERATED OPTIMIZATION

**source**: ICLR 2021

这个真的抽象，就是server端，自适应选择优化器 adam这种。



---

2024/7/13

**title**: WHAT DO WE MEAN BY GENERALIZATION IN FEDERATED LEARNING?

**source**: ICLR 2022

 participation gaps can quantify client dataset heterogeneity. 讨论泛化能力。以及提出一些重要的指标。



----

2024/7/14

title:  BAYESIAN CORESET OPTIMIZATION FOR PERSONALIZED FEDERATED LEARNING

source: ICLR 2024

需要了解贝叶斯核心集优化。



title: FEDERATED LEARNING VIA POSTERIOR AVERAGING: A NEW PERSPECTIVE AND PRACTICAL ALGORITHMS

source: ICLR 2021

使用蒙特卡洛采样拟合后验，然后用这些来作为聚合的权重。



---

2024/7/20

title：SIMPLE MINIMAX OPTIMAL BYZANTINE ROBUST AL GORITHM FOR NONCONVEX OBJECTIVES WITH UNI FORM GRADIENT HETEROGENEITY

source：ICLR 2024

好！建议精读！



---

2024/7/24

title:  Aligning model outputs for class imbalanced non‑IID  federated learning

source:  [Aligning model outputs for class imbalanced non-IID federated learning | Machine Learning (springer.com)](https://link.springer.com/article/10.1007/s10994-022-06241-5)

建议精读。



---

title: FedSampling: A Better Sampling Strategy for Federated Learning

source:  arXiv:2306.14245v1

具有参考价值，针对训练样本的公平性，采用的计算方法都是基于伯努利分布。



---

2024/7/23

title: Federated Learning under Heterogeneous and Correlated Client Availability

source: DOI: 10.1109/INFOCOM53939.2023.10228876

理论性很强。需要细读







