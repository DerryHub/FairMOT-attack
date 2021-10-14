# outline

* 确定攻击代表性模型
* overview of FairMOT
  * detection branch
  * re-ID branch
  * online association
* problem definition
* single-target attack
* multiple-targets attack
* implementation details

# method

**ID-Exchanged Attack**

In this section, we propose a noval method named ID-exchanged attack for multiple object tracking based on combination of detection and matching. Our method aims to make the id of the target object different from the original by exchange the ids between the target object and its adjacent object.  Representatively, we choose FairMOT[1] as our target being attacked due to its high popularity among in real-time MOT system. Besides, it is easy for our method to be implemented in other MOT model based on combination of detection and matching[2,3,4].

## Overview of FairMOT

Achieving a good balance between accuracy and speed, FairMOT gets extremely popular in academia and industry. The fairness between the tasks of object detection and matching allows FairMOT to obtain high performance on MOT challenge datasets[5,6,7,8]. As shown in **Figure 1**, the network architecture of FairMOT consists of two homogeneous branches to predict the location of object and re-ID features. Besides the branchs, online association is also the important component of FairMOT.

**Detection Branch.** The anchor-free detection branch of FairMOT is built on CenterNet, which predicts three heads: heatmap head, box offset head and size head. Heatmap head learns the locations of the object centers. Specifically, the dimension of the heatmap is $1\times H\times W$. We define bounding box of the $i$-th object in $t$-th frame as $box^t_i=(x_1^{t,i},y_1^{t,i},x_2^{t,i},y_2^{t,i})$. And then the object center can be respectively computed $(c_x^{t,i},c_y^{t,i})$ as $c_x^{t,i}=\frac{x_1^{t,i}+x_2^{t,i}}{2}$ and $c_y^{t,i}=\frac{y_1^{t,i}+y_2^{t,i}}{2}$. The location of the bounding box on the heatmap can be obtained by dividing the stride (4 in FairMOT) $(\lfloor\frac{c_x^{t,i}}{4}\rfloor,\lfloor\frac{c_y^{t,i}}{4}\rfloor)$. The value of heatmap means the confidence score that there is an object center at the corresponding location. Also, the goal of box offset head is to localize objects more precisely. Denote the offset and size heads in $t$-th frame as $offset^t\in\R^{2\times H\times W}$ and $size^t\in\R^{2\times H\times W}$. Similarly, the GT offset head is computed as $offset_i^t=(\frac{c_x^{t,i}}{4}-\lfloor\frac{c_x^{t,i}}{4}\rfloor,\frac{c_y^{t,i}}{4}-\lfloor\frac{c_y^{t,i}}{4}\rfloor)$ and the GT size head is computed as $size_i^t=(x_2^{t,i}-x_1^{t,i},y_2^{t,i}-y_1^{t,i})$. 

**Re-ID Branch.** Re-ID branch learn the difference of objects. Denote the feature map as $feature^t\in\R^{512\times H\times W}$. The re-ID feature $feature^t_{x,y}\in\R^{512}$ means the feature vector, whose $L_2$ norm equals to 1, of an object centered at $(x,y)$. Then cosine similarities of the features are computed in matching module to evaluate the similarity of objects.

**Oline Association.** FairMOT follows the standard online tracking algorithm[2] to association boxes. In the first frame, we initialize tracklets from predicted bounding boxes and features. Then in each next frame, detected boxes are linked to the existing tracklets according to their cosine distance and their box distance.

## Problem Definition



# references

[1] FairMOT

[2] JDE

[3] CenterTrack

[4] ChainedTracker

[5] MOT15

[6] MOT16

[7] MOT17

[8] MOT20
