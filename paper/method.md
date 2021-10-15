# outline

* 确定攻击代表性模型
* overview of FairMOT
  * detection branch
  * re-ID branch
  * online association
* problem definition
  * 定义数据
  * 定义模型及输出
  * 定义任务
    * single-target attack
    * multiple-targets attack
* single-target attack
* multiple-targets attack
* implementation details

# method

**ID-Exchanged Attack**

In this section, we propose a noval method named ID-exchanged attack for multiple object tracking based on combination of detection and matching. Our method aims to make the id of the target object different from the original by exchanging the ids between the target object and its adjacent object.  Representatively, we choose FairMOT[1] as our target model being attacked due to its high popularity among in real-time MOT system. Besides, it is easy for our method to be implemented in other MOT model based on combination of detection and matching[2,3,4].

## Overview of FairMOT

Achieving a good balance between accuracy and speed, FairMOT gets extremely popular in academia and industry. The fairness between the tasks of object detection and matching allows FairMOT to obtain high performance on MOT challenge datasets[5,6,7,8]. As shown in **Figure 1**, the network architecture of FairMOT consists of two homogeneous branches to predict the location of object and re-ID features. Besides the branchs, online association is also the important component of FairMOT.

**Detection Branch.** The anchor-free detection branch of FairMOT is built on CenterNet, which predicts three heads: heatmap head, box offset head and size head. Heatmap head learns the locations of the object centers. Specifically, the dimension of the heatmap is $1\times H\times W$. We define bounding box of the $i$-th object in $t$-th frame as $box^t_i=(x_1^{t,i},y_1^{t,i},x_2^{t,i},y_2^{t,i})$. And then the object center can be respectively computed $(c_x^{t,i},c_y^{t,i})$ as $c_x^{t,i}=\frac{x_1^{t,i}+x_2^{t,i}}{2}$ and $c_y^{t,i}=\frac{y_1^{t,i}+y_2^{t,i}}{2}$. The location of the bounding box on the heatmap can be obtained by dividing the stride (4 in FairMOT) $(\lfloor\frac{c_x^{t,i}}{4}\rfloor,\lfloor\frac{c_y^{t,i}}{4}\rfloor)$. The value of heatmap means the confidence score that there is an object center at the corresponding location. Also, the goal of box offset head is to localize objects more precisely. Denote the offset and size heads in $t$-th frame as $offset^t\in\R^{2\times H\times W}$ and $size^t\in\R^{2\times H\times W}$. Similarly, the GT offset head is computed as $offset_i^t=(\frac{c_x^{t,i}}{4}-\lfloor\frac{c_x^{t,i}}{4}\rfloor,\frac{c_y^{t,i}}{4}-\lfloor\frac{c_y^{t,i}}{4}\rfloor)$ and the GT size head is computed as $size_i^t=(x_2^{t,i}-x_1^{t,i},y_2^{t,i}-y_1^{t,i})$. 

**Re-ID Branch.** Re-ID branch learn the difference of objects. Denote the feature map as $feature^t\in\R^{512\times H\times W}$. The re-ID feature $feature^t_{x,y}\in\R^{512}$ means the feature vector, whose $L_2$ norm equals to 1, of an object centered at $(x,y)$. Then cosine similarities of the features are computed in matching module to evaluate the similarity of objects.

**Oline Association.** FairMOT follows the standard online tracking algorithm[2] to association boxes. In the first frame, tracklets from predicted bounding boxes and features are initialized. Then in each next frame, detected boxes are linked to the existing tracklets according to their cosine distance and their box distance.

## Problem Definition

Denote video $V=\{I_1,\dots,I_{t-1},I_t,I_{t+1}\dots,I_N\}$​​​​ that contains $N$​ video frames. In simple terms, there are two intersectant tracking objects predicted from the tracker $f_\theta(\cdot)$​ in the video. Respectively, denote the two predicted tracking objects $T_i=\{O_{s_i}^i,\dots,O_t^i,\dots,O_{e_i}^i\}$​ and $T_j=\{O_{s_j}^j,\dots,O_t^j,\dots,O_{e_j}^j\}$​ where they are adjacent at $t$​​​​​​​​​-th frame. Similarly, we can define the bounding boxes and features of the tracking objects as $B_k=\{box_{s_k}^k,\dots,box_t^k,\dots,box_{e_k}^k\}$​ and $F_k=\{feat_{s_k}^k,\dots,feat_t^k,\dots,feat_{e_k}^k\}$​ where $k\in\{i,j\}$​​​. In general, $box\in\R^4$ and $feat\in\R^{512}$​​​ are used to represent the bounding boxes and features of tracking objects in FairMOT.

Specifically, the tracker $f_\theta(\cdot)$​​ contains detection branch, re-ID branch and online association. At the $t$​​​-th video frame, we can obtain bounding box $box_t^k$​​​ and feature $feat_t^k$​​ of tracking object $O_t^k$​​​. For the next frame, we need to compute the similarity of the obejects predicted and the tracklet pool at the $t$​​​​-th frame. The similarity contains bounding boxes distances and features distances. To be specific, distances of bounding boxes are computed as $d^{ij}_{box}=Dis_{box}(K(box_t^i),box_{t+1}^j)$​​​ and $d_{feat}^{ij}=Dis_{feat}(smooth(feat_t^i),feat_{t+1}^j)$​​ where $K(\cdot)$​​ means the Kalman filter and $smooth(feat_t^i)=\alpha \cdot smooth(feat_{t-1}^i)+(1-\alpha)\cdot feat_t^i$​​​​​. Then the linear assignment problem is solved by Hungarian algorithm with the final cost matrix between tracking objects in tracklet pool and objects predicted at the next frame $C=\{\lambda d_{box}^{ij}+(1-\lambda)d_{feat}^{ij}\}$​​​.

Denote the adversarial video $\hat V=\{I_1,\cdots,I_{t-1},\hat I_t,\cdots,\hat I_{t+n-1},I_{t+n},\cdots,I_N\}$​ where $I$​ means the original frame and $\hat I$​​​ means frame with tiny disturbance. Here are the definitions of single-target attack and multiple-targets attack in MOT below:

1. Single-Target Attack. As for a tracker $T_i$​​, $T_j$​​ is another tracker adjacent with $T_i$​​ at $t$​​-th frame. The adversarial video $\hat V$​​ guides the tracker $\hat T_i=\{O_{s_i}^i,\dots,O_{t-1}^i,O_t^j,\dots,O_{t+n-1}^j,O_{t+n}^j,\dots,O_{e_j}^j\}$​​. Only $n$​​​, a minimum of 1 frame is required, frames attacked, the attack method can make $\hat T_i$​​ error​​ and maintain the status after being attacked.
2. Multiple-Targets Attack. Similarly, the adversarial video $\hat V$​​ causes all trackers that are adjacent with others error as far as possible. In general, the method do not need all frames of the video to be attacked as same as single-target attack.

# references

[1] FairMOT

[2] JDE

[3] CenterTrack

[4] ChainedTracker

[5] MOT15

[6] MOT16

[7] MOT17

[8] MOT20
