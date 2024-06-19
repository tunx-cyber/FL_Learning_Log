Enabling Communication-Efficient Federated Learning via Distributed Compressed Sensing  

DOI: 10.1109/INFOCOM53939.2023.10229032  

sparse:稀疏的

bandwidth-limited, thus gradient compression techniques such as sparsification and quantization have been introduced to reduce transmitting bits in each communication round.

- The main idea of sparsification is dropping out the least significant elements as most entries in the gradients of DNN are approaching zero.

- Quantization methods focus on using fewer bits to represent gradients in communication while maintaining inference accuracy of the converged model with minimal loss

Recently, gradient compression based on compressed sensing (CS) [4]–[6] has been put forward as an orthogonal technique that can be combined with the aforementioned two methods to achieve a higher compression rate.   

In CS theory, sparse high-dimensional signals can be precisely reconstructed with an overwhelming probability from a few random measurements far lower than the Shannon-Nyquist sampling rate》

这里的round是每次更新

Exploiting:利用 correlations:相关性 spatially:在空间上

correlations of gradients between clients

gradient correlations between adjacent rounds definitely exist, correlations are much stronger than those between clients  

The existing CS-based compression techniques emphasis on compressing the intra-redundancy in an individual gradient yet overlooking the inter-redundancy of gradients between clients or adjacent rounds that takes forms of the spatio-temporal(时空) correlations, which still leaves plenty of room for improvements  

this paper adopts distributed compressed sensing (DCS) [16] to compress gradients, leveraging gradient correlations between temporally adjacent rounds to realize a deeper compression. As the intersection between distributed source coding [17] and CS theory [9]

Nonetheless：尽管如此 infeasible：不可行 alleviate：减轻 side information：附带信息 calibration ：校准，标定。 adverse：不利的

 supplementary：增补的

pertinent：有关的

Compressed Sensing：压缩传感

ensuing：随之而来的

catastrophic domino effect：灾难性多米诺骨牌效应

- 疑惑：signal是什么，(类似影像压缩？相邻的frame之间相关性很大)
- hard thresholding pursuit (HTP)  是什么



top-k sparsification before CS;  



SVDFed: Enabling Communication-Efficient Federated Learning via Singular-Value-Decomposition  

Haolin Wang  

are quantization [2]–[4] and sparsification [5], [6]. The former reduces the number of bits required to represent each gradient value, and the latter only selects a portion of the gradients.   

This technique is based on the following observation that the elements within a client’s gradients are usually correlated [8]– [10]. Correspondingly, low-rank approximation first converts the gradients of a client into a matrix form, and then decomposes the matrix into the product of two much smaller matrices [11], [12]. Transmitting the two matrices is much more communication efficient than transmitting the original matrix  

More recently, some researchers find that, in traditional dataparallel distributed training, gradients in different rounds are correlated. Then they propose a technique called as learned gradient compression (LGC) [13]. LGC has a ‘warm-up’ stage, in which full gradients of clients in initial rounds are transmitted to the server to train a compressor. Clients then utilize the compressor to compress their gradients afterwards  





Tackling System Induced Bias in Federated Learning: Stratification and Convergence Analysis  

Ming Tang

state-of-the-art：最先进的

heterogeneity：不匀称性

incorporate:混合

stratified：分层的

aggregated：聚合的





A Reinforcement Learning Approach for Minimizing Job Completion Time in Clustered Federated Learning  

Ruiting Zhou  

straggler：掉队者

mitigating：减轻

Empirical：经验主义的

First, clients have different device usage patterns and data samples, leading to different data distributions located on devices. It has been shown that training the FL model on non-IID data will greatly reduce the accuracy of the FL model and increase the model convergence time [4], [5], [6]. Second, in practice, the resources possessed by clients are likely to be different, resulting in heterogeneity of computing power and communication capabilities. The device heterogeneity dictates that the FL job completion time is constrained by the slowest client under the synchronous framework, which is known as the straggler effect [7]. Under wireless and mobile environment with limited connectivity, the straggler effect becomes the main bottleneck in realizing FL  



i) Pre-clustering stage. We capture the device heterogeneity of clients, and model a maximum intra-cluster variance minimization problem to eliminate the stragglers within a cluster. ii) Training stage. By considering both non-IID data and the straggler effect among clusters, we model the job completion time minimization problem under the clustered FL framework  



we use a DRL algorithm to take two actions: i) learn the optimal number of iterations for intracluster aggregation for each cluster, such that each cluster has similar time of one round inter-cluster aggregation (i.e., the computation time for obtaining the updated cluster model plus the communication time to send the cluster model to the cloud server). As a result, the straggler effect among clusters can be mitigated; ii) select clients who contribute the most to the model accuracy by learning the non-IID attributes of clients. The impact of non-IID data can be reduced by the intelligent selection of clients  



Network Adaptive Federated Learning: Congestion and Lossy Compression  

Parikshit Hegde  





A Data Privacy Protection Scheme Integrating Federal Learning and Secret Sharing  

diversified：多元化

explode：爆炸

cope：对抗，处理

prominent：重要的

fuse：结合

tremendous：巨大的

blockchain 

Primitive Root: 本源根

抄袭diffie-hellman



Joint Superposition Coding and Training for Federated Learning over Multi-Width Neural Networks  

heterogeneous:各种各样的

superposition  ：叠加

address：设法解决，演讲，称呼

hinge: 铰链

hinge on：依赖于

synergy ：协同作用

prune  ：修剪

knowledge distillation  ：知识提炼

incurs：引起

看不懂，好复杂

非独立同分布（Non-IID）是一个在统计学和机器学习中常用的概念，它描述的是样本数据不满足独立同分布（IID）的假设。在IID假设下，样本数据是相互独立的且来自相同的概率分布。然而，在实际情况中，这种假设往往是不符合实际的，样本数据可能具有相关性或者来自不同的概率分布，这就构成了非独立同分布的情况。

以协同过滤（Collaborative Filtering）为例，这是一个常见的机器学习应用。在这个应用中，用户对物品的评分数据通常不满足独立同分布假设。因为同一用户对不同物品的评分往往具有一定的关联性，即用户的评分行为不是独立的。此外，不同用户可能因为个人喜好、文化背景等因素，对相同物品的评分也会有所不同，这导致了评分数据来自不同的概率分布。

再举一个例子，假设我们有一个全球性的电商平台，不同地区的用户由于文化、习惯、经济状况等因素的差异，对商品的购买偏好也会有所不同。这些购买数据就构成了非独立同分布的数据。例如，某些商品可能在某些地区非常受欢迎，而在其他地区则不受欢迎。因此，如果直接基于这些非独立同分布的数据进行模型训练，可能会导致模型在某些地区的预测效果较差。

处理非独立同分布数据的方法有很多，比如可以考虑使用联合建模、混合建模等技术来应对数据之间的关联性，或者通过数据预处理、特征工程等手段来减少数据分布的不一致性。在联邦学习的上下文中，也可以通过设计特殊的模型结构和优化算法来适应不同参与方上的非独立同分布数据。
