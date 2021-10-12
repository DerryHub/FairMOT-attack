# 3W2H

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

* 介绍以往工作
  * 简述第一篇单目标跟踪对抗样本，for example。
  * 然而，对于多目标跟踪的对抗样本，还没得到足够重视。简述多目标跟踪对抗样本内容。
  * 对抗样本的研究既能帮助研究者更深入理解模型，提升其鲁棒性，然而在多目标跟踪领域重视程度不够。
* 引到我们的工作
  * 然而，以往在目标跟踪上的对抗样本都是基于检测框的攻击。
  * 然而，在基于深度学习的单步MOT上，是一个首先定位，再用特征和位置信息来匹配的模型，有一个完备的后处理。
  * 在这个后处理中，
* in this paper
  * 我们是第一篇在基于深度学习单步MOT提出攻击对抗任务的文章。
  * 我们发现在跟踪视频中存在特别脆弱的点，可以只在几帧添加极小的噪声就可以攻击成功。
  * 对此我们提出了一种完备的基于目标框和其特征的攻击任务及方法，可以非常隐蔽地使得目标跟错。
  * 



简单介绍以往工作

根据以往工作得到的启发

简述为什么要做这个工作

从原理上分析这个工作


本文贡献

* 引出MOT，简要介绍用途；攻击该系统的意义，必要性
* 简单介绍前人工作，（做法，不足）
* 引出本文工作
  * 对MOT系统的攻击类型
    * 基于detection
    * 基于feature
    * both
  * 针对于第三种类型提出的方法
* contribution
  * 总结针对MOT系统的攻击类型：对目标框和特征
  * 提出一种高效的攻击方法：ID交换
  * 实验结果对比得到该攻击方法的高效，可行性


# Related Work

目标检测的对抗样本

mot背景

# Method



# Experiments



# Discussion



# Conclusion





* 对抗攻击的研究既能帮助研究者加深对模型决策机理的理解，也能为设计更加鲁棒的算法提供思路。但是目前，对抗攻击在目标跟踪领域尚未引起足够的重视
