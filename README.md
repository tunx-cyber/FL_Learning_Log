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

