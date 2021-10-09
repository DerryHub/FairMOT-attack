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

简单介绍以往工作

根据以往工作得到的启发

简述为什么要做这个工作

从原理上分析这个工作

本文贡献

# Related Work

目标检测的对抗样本

mot背景

# Method



# Experiments



# Discussion



# Conclusion

