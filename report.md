# 主要工作为文献调研。

## 个人总结

### 1、对于个性化联邦学习：

很多策略都是找相似。比如简化欧氏距离，使用拉格朗日数乘法求解最优解，使用多做一次训练的参数作为指导。

### 2、传统联邦学习

目前很多论文（就我所研读）主要是靠**知识蒸馏**减少开销，增加训练效率。知识蒸馏的过程中使用到了proxy data。但是这种方法面临缺少一个很好的teacher model，因此有一种算法提出在客户端通过**density-ratio estimation**进行筛选，过滤out of distribution的sample，服务端也对一些模糊的知识蒸馏进行过滤。通过两个层面的选择，最终提升了FL的效率[1]。除此之外，[2]结合了GAN和知识蒸馏来提升FL的的收敛效率。[10]结合知识蒸馏，使用一个buffer，记录历史模型，对历史模型进行平均聚合。[26]使用parameter-efficient finetuning (PEFT) method + adapter & Mutual Knowledge Distillation。[27]尝试生成客户端数据集的补集来进行知识蒸馏。

很多文章主要集中于**具体应用场景**中。主要方向有针对能源cost的schedule，针对网络方面和网络质量方面的带宽限制提出的算法（原型+softpool+practical byzantine fault-tolerance[4]，简化scaffold公式，通过数学处理，减少了上传量，以及采用了动量的处理[16]），也有针对隐私安全问题提出具体的算法[3]。不过本人对这些方向很迷惑，很多公式闻所未闻，过于陌生，目前兴趣点不在这里。这些文章在INFOCOM上面很多，主要结合的方向就是计算机网络。

有的文章就是结合**cluster**。[5]考虑了一个edge或者cluster的重要性，设计了**LB-Net**减少通信量。[25]使用type prompt + cluster + Gaussian Mixture Model + GC-Net。研究了[28]minority samples。[31]结合cluster和强化学习解决straggler effect。

有的文章就是**研究梯度**，对model的parameters进行一些比较令人迷惑的操作。[8]这篇文章使用了变异，改变梯度的方向摆脱局部最优，不过没有理论证明，可行性有待商榷，不过可以作为参考，受这篇文章的启发，我想的是使用遗传算法来解决。有的文章就是拼接一些神经网络层也可以获得好的效果。有的文章对动量情有独钟。[16] [18]就研究了动量。也有缝合怪，[21]就是动量加客户端选择。[30]按照一种特殊比例混合了局部模型和全局模型。这种比例利用了gram matrix，不过为了减少gram matrix的运算量，使用最后一层来计算gram matrix。👍

有的文章是结合**强化学习**，个人觉得有点暴力解法，因为这种方法极其不稳定，训练结果很多时候都是靠运气。[23] 利用强化学习找到比较好的batch size解决训练能力不同的客户端的问题，[9]结合了auction算法，可以作为一种方向的参考。既然都是用RL了，那肯定有人就想着做多目标优化，[11]这篇文章就是多目标优化的FL，不过其实整个过程其实非常公式化，创新点少。个人对强化学习不看好。[29]不仅多目标还多任务。

有的文章对于**数据异质性研究**很深入，我非常喜欢，[6]这篇联盟FL分析了异构型对于model的准确性的关系，除此之外我也找到了[7]这篇比较早提出联盟这种概念的文章，很赞同里面的一些对于联盟形成的idea。👍

有的文章就是结合一些**概率论**的观点。[14]使用贝叶斯核心集进行优化。[15]使用蒙特卡洛采样拟合后验，然后用这些来作为聚合的权重。[16]针对训练样本的公平性，采用的计算方法都是基于伯努利分布。[17]利用时间先验，设计了多分支网络，使用聚类模型的联合训练，针对周期性的联邦学习场景。[20]利用后验推断。[22]意图使用近似的分布解决了non-iid的泛化问题。也对神经网络进行了分层。非常值得借鉴。👍

有的文章就**结合很多领域**。比如集成学习[12]。持续学习[19]，GAN[24]。

有的文章比较**抽象**。[13]在服务端自适应选择优化器，就是我们常用的adam这种。[32]使用符号推理，这篇比较难懂，目前我也没怎么懂。

除此之外有很多其他类型的文章，这里很难归为以上的类别，具体参考我的阅读记录https://github.com/wuweipower/FL_Learning_Log

### 一些感悟

1. 很多论文感觉并未做出特别明显的，或者说，很创新的贡献，但是却能发表，我觉得主要原因就是使用结合的方法：将FL与一些应用场景结合，做一些普遍的理论推导和实验。而这些应用场景常常都是机器学习中的其他领域。总的来说就是其他领域的问题加上FL就感觉可以发论文了。不过这个缺少创新性，如果作为自己硕士论文答辩，底气不足，思维不够开阔。比如强化学习加FL，我看到太多了，论文比较公式化，创新的地方主要是公式，不过我觉得很多没有抓住问题的核心。
1. 很多文章的公式证明篇幅比较大，生涩难懂，对我来说特别陌生。不过依据师兄的说法这些很多都是引用，因此我对于其中一些文章做了标记，为未来的文章编写提供参考。
1. 过程中学习到了很多新知识，很充实，但是感觉很多东西特别重复，有阅读疲劳。

### 一些不成熟的思考

将FL作为优化搜索问题，往演化算法方向靠，并且研究**梯度**，搜索梯度，去获得一个很好的解。以下是我的一些算法核心思想。

```c++
1. select 根据调研的论文，从客户端选择入手这是比较常见的。我的想法是以贡献值进行选择并且是以某种概率选择。贡献值又该如何定义？（数据集大小？数据新鲜度？模型受训练新鲜度？parameter抖动的大小？）我们可以参考shaply value，以聚合这个客户端的参数对性能提升的多少作为具体贡献值。不过计算量特别大，另辟蹊径还是进行改良？
2. mutate 参考"FedMut: Generalized Federated Learning via Stochastic Mutation"这篇文章的想法，对于gradient或者parameters进行变异，摆脱局部最优。或者我们直接把gradient作为gene。我们该怎么变异？主要是有启发式的变异才比较好，增加算法效率。以什么作为启发？这是个问题。除此之外，变异的具体操作可以参考动量操作
```



### 未来工作安排

1. 多看看论文代码，主要学习对于神经网络中的参数的处理。
2. 学习机器学习的基本公式（我发现我对很多论文中出现的公式以及变量不太了解）
3. 着手我的一些想法，并且寻找更多与我思考方向相似的文章。
4. 对于我自己标注👍的方向的文章，会重点多读。



## 文献参考

以下是参考文献，为了减少篇幅，只列出文章标题。

文章的具体的来源，可以参考我的GitHub：https://github.com/wuweipower/FL_Learning_Log

[1] Selective Knowledge Sharing for Privacy-Preserving Federated Distillation without A Good Teacher

[2] Towards Data-Independent Knowledge Transfer in Model-Heterogeneous Federated Learning 

[3] PrivAim: A Dual-Privacy Preserving and Quality-Aware Incentive Mechanism for Federated Learning 

[4] FedRFQ: Prototype-Based Federated Learning With Reduced Redundancy, Minimal Failure, and Enhanced Quality  

[5] User-Distribution-Aware Federated Learning for Efficient Communication and Fast Inference

[6]  Coalitional FL: Coalition Formation and Selection in Federated Learning with Heterogeneous Data

[7] A Coalition Formation Game Approach for Personalized Federated Learning

[8]  FedMut: Generalized Federated Learning via Stochastic Mutation

[9] Competitive-Cooperative Multi-Agent Reinforcement Learning for Auction-based Federated Learning

[10] FEDGKD: Toward Heterogeneous Federated Learning via Global Knowledge Distillation

[11] FairTrade: Achieving Pareto-Optimal Trade-Offs between Balanced Accuracy and Fairness in Federated Learning

[12] Boosting with Multiple Sources

[13] ADAPTIVE FEDERATED OPTIMIZATION

[14] BAYESIAN CORESET OPTIMIZATION FOR PERSONALIZED FEDERATED LEARNING

[15] FEDERATED LEARNING VIA POSTERIOR AVERAGING: A NEW PERSPECTIVE AND PRACTICAL ALGORITHMS

[16] Stochastic Controlled Averaging for Federated Learning with Communication Compression

[17] DIURNAL OR NOCTURNAL? FEDERATED LEARNING OF MULTI-BRANCH NETWORKS FROM PERIODICALLY SHIFTING DISTRIBUTIONS

[18] MOMENTUM BENEFITS NON-IID FEDERATED LEARNING SIMPLY AND PROVABLY

[19] ACCURATE FORGETTING FOR HETEROGENEOUS FEDERATED CONTINUAL LEARNING

[20]  FEDERATED LEARNING VIA POSTERIOR AVERAGING: A NEW PERSPECTIVE AND PRACTICAL ALGORITHMS

[21] FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection

[22] FEDIMPRO: MEASURING AND IMPROVING CLIENT UP DATE IN FEDERATED LEARNING

[23] Adaptive Federated Learning on Non-IID Data With Resource Constraint

[24] Federated Generative Model on Multi-Source Heterogeneous Data in IoT

[25] FedGCR: Achieving Performance and Fairness for Federated Learning with Distinct Client Types via Group Customization and Reweighting

[26] FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning

[27] FAKE IT TILL MAKE IT: FEDERATED LEARNING WITH CONSENSUS-ORIENTED GENERATION

[28] LIKE OIL AND WATER: GROUPROBUSTNESS METHODS AND POISONING DEFENSES MAY BE AT ODDS

[29] Federated Multi-Objective Learning

[30] HETEROGENEOUS PERSONALIZED FEDERATED LEARNING BY LOCAL-GLOBAL UPDATES MIXING VIA CONVERGENCE RATE

[31] A Reinforcement Learning Approach for Minimizing Job Completion Time in Clustered Federated Learning

[32] Formal Logic Enabled Personalized Federated Learning through Property Inference