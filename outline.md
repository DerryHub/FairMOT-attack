#  3W2H

## why

* motivation
  * MOT应用越来越广，越来越火
  * MOT的跟踪失败会影响目标的位置、速度以及移动方向，可能造成严重后果
    * 监控、自动驾驶
  * 发现MOT脆弱点，只需要攻击几帧即可使目标跟踪失败

## what

* contribution
  * 发现一种简易的攻击方法，能够轻松攻击MOT中的目标
  * 使目标跟踪失效的方式：检测失败、特征失效、ID交换
  * why ID交换？MOT后处理会预防检测失败（max_time_lost）和特征失效（smooth alpha）。
  * ID交换高效并健壮

## how

* theorem

  * MOT后处理分析

* model

  * ？ attack method

* algorithm

  * define $D_i^t,D_j^t,F_i^t,F_j^t$。第$t$帧第$i/j$目标的框及特征

  * 特征优化

    * $$
      \min_\hat{F}CosineSimilary(Smooth(F_i^{t-1}),\hat{F_j^t})+CosineSimilary(Smooth(F_j^{t-1}),\hat{F_i^t})\\
      \max_\hat{F}CosineSimilary(Smooth(F_i^{t-1}),\hat{F_i^t})+CosineSimilary(Smooth(F_j^{t-1}),\hat{F_j^t})
      $$

  * 目标框优化

    * $$
      \min_\hat{D}Distance(Center(Kalman(D_i^{t-1})),Center(\hat{D_j^t}))+Distance(Center(Kalman(D_j^{t-1})),Center(\hat{D_i^t}))\\
      \min_\hat{D^t}FocalLoss(\hat{D^t})\\
      \min_\hat{D}MSE(WH(\hat{D^t}),WH(D^t))\\
      \min_\hat{D}MSE(reg(\hat{D^t}),reg(D^t))
      $$
  
  * 噪声优化
  
    * $$
      \min_\hat{X}(\hat{X}-X)^2
      $$
  
  * 多目标攻击：难样本

## how much

* experiment
  * 成功率：接近100%
  * 攻击帧数：最少1，平均、中位数4左右
  * L2距离：最小<2，最大10左右
* discussion

## what then

* conclusion
  * 目前MOT非常脆弱
  * 需要加强MOT强度
  * 找视频最脆弱的地方可以使视频模型失效

# Introduction

* 介绍MOT
  * 简单说明MOT重要性及其应用场景
  * 介绍MOT数据集：MOT challenge、PathTrack等（x）
  * 简单介绍目前MOT方法
    * 基于Tracking-by-detection的MOT：SORT、DeepSORT
    * 基于检测和跟踪联合的MOT：JDE、FairMOT、CenterTrack、ChainedTracker等
    
    Figure 1：上述两种MOT的概念图
  * 简单说明第二种方法更受关注
* 简单介绍对抗样本
  * 介绍各个领域对抗样本：分类、目标检测、单目标跟踪、多目标跟踪
* 引出我们的方法
  * 然而，在基于检测和跟踪联合的MOT上的对抗样本还不受关注
  * 在本文我们针对基于检测和跟踪联合的MOT上的一些代表性模型进行攻击
  * 但是其有完备的匹配策略，难以攻击
    * 卡尔曼
    * 平滑特征
  * for example，对框进行简单攻击使得其检测不到目标，需要连续攻击数十帧才能欺骗成功，如实验所示
  * 对此，我们提出了ID交换的攻击手段，可以在最少只需要攻击一帧，平均几帧的情况下对单个特定目标攻击成功
  * 据我们所知，这是第一篇针对基于检测和跟踪联合的MOT及其后处理的攻击
* contribution
  * 我们找到了MOT视频中最脆弱的区域，在该区域只需要极小的扰动便可以对某一个目标的轨迹产生影响
  * 提出了一种针对基于检测和跟踪联合的MOT及其后处理的高效的和新奇的攻击方法。
  * 实验证实了我们攻击手段的有效性，以及MOT和后处理的弱点

-----------------------------------------------------*-----------------------------------------------------------

* 基于深度学习CNN的Mot 模型逐渐成熟，在很多应用上取得excellent perform。
  * what is Mot
    * how it work：。。。
    * what is its goal：correctly tracking multiple object 
      * find and locate object on each frame
      * correctly link the object to existing tracks
  * what is used for
  * popular algorithm
* 然而自对抗样本被学习以来，发现很多sota 模型容易被图形上轻微的扰动影响（巨大），那么对于这些成熟的Mot模型框架是否也存在同样的问题
  * first，what does it mean to attack the MOT system
    * As methion above the goal of MOT is correctly tracking multiple object 
    * so the goal of attacking MOT is to make it uncorrectly tarcking multiple object
      * how do
        *  make locate uncorrctly or even worse make it unable to  find the object 
          * previous work ...
        * uncorrectly link the object to existing tracks
          * previous work .....
* the previous works are mainly focus in disturbing the object detection ，the adversarial  method of making it uncorrectly identify track still lack of explore，in our work 。。。。。。
  * 提出方法
* contirbution：
  * first explict the MOT attacking task
  * explore an novel 。。。。。
    * efficiency 。。。
    * mot 尚未成熟的后处理流程
  * expriment 。。。。


# Related Work

* MOT
  * MOT是做什么的
  * 简单介绍目前MOT方法
    * 基于Tracking-by-detection的MOT：SORT、DeepSORT
    * 基于检测和跟踪联合的MOT：JDE、FairMOT、CenterTrack、ChainedTracker等
  * 具体介绍基于Tracking-by-detection的MOT
  * 具体介绍基于检测和跟踪联合的MOT
* AE
  * 简述AE的历史和意义
  * 介绍基于梯度下降的AE算法，FGSM、IFGSM等等

# Method

* in this section, we do ...

* overview of fairmot

  Figure 2：network archtecture of fairmot

* problem definition

  Figure 3：ID交换示意图

* single attack

  Algorithm 1

  Figure 4：center closing

* multiple attack

  Algorithm 2

* implementation details

# Experiments

* 实验背景
* 评估方式
  * 攻击成功率
  * 攻击帧数
  * L2距离
* 对feature和detection的单独攻击效果
  * Figure 5：攻击效果图
* ID交换攻击
  * single
    * Table 1：每个数据集在每个模型上的攻击效果
  * multiple
    * Table 2：每个数据集在每个模型上的攻击效果

* model：FairMot，JDE...

* train dataset: MOT 15 ，MOT 20， CalTech， CUHK-SYSU，prw

* val dataset：MOT17,MOT 16, MOT15,MOT20

  * 计算原始预测成功的轨迹：10 -inter -20

  *  single表格：

    * 含原始指标数据集

    | Model | 数据集 | 原始跟踪成功率（成功数量/满足条件数量） | 攻击后（成功数量/满足条件数量） | drop | l2   | avg attack frame （攻击帧数总和/满足条件id数总和） | 满足条件id数量 |
    | ----- | ------ | --------------------------------------- | ------------------------------- | ---- | ---- | -------------------------------------------------- | -------------- |
    |       |        |                                         |                                 |      |      |                                                    |                |

    * 无原始标注

      | model | dataset | success rate | l2   | avg attack frame | ids  |
      | ----- | ------- | ------------ | ---- | ---------------- | ---- |
      |       |         |              |      |                  |      |

    

* mulitple

  * 含原始指标数据集

  | model | 数据集 | 原始跟踪成功率（成功数量/满足条件数量） | 攻击后（成功数量/满足条件数量） | drop | l2   |      | 满足条件id数量 |
  | ----- | ------ | --------------------------------------- | ------------------------------- | ---- | ---- | ---- | -------------- |
  |       |        |                                         |                                 |      |      |      |                |

  * 无原始标注

  | model | dataset | success rate | l2   | ids  |
  | ----- | ------- | ------------ | ---- | ---- |
  |       |         |              |      |      |

  

* FairMot 对比实验
  * 有无center leaping 
  * 有无难样本（multiple）
  * 攻击帧数（1，2，，，）与攻击准确率（x %）（single） ourmethod / 隐身 single center leaping
  *  iou thr



# Discussion



# Conclusion





* 对抗攻击的研究既能帮助研究者加深对模型决策机理的理解，也能为设计更加鲁棒的算法提供思路。但是目前，对抗攻击在目标跟踪领域尚未引起足够的重视
