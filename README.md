# TraSw for FairMOT

* A Single-Target Attack example (Attack ID: 19; Screener ID: 24):

<table>
    <tr>
        <td ><center><img src="assets/original.gif" ><b> Fig.1  Original </b> </center></td>
        <td ><center><img src="assets/attacked.gif" ><b> Fig.2  Attacked </b> </center></td>
    </tr>
</table>

> [**TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**](https://arxiv.org/abs/2111.08954),
> Delv Lin, Qi Chen, Chengyu Zhou, Kun He,
> *[arXiv 2111.08954](https://arxiv.org/abs/2111.08954)*

## Abstract

Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack.

# Installation

* same as [FairMOT](https://github.com/microsoft/FairMOT)

# Data preparation

* same as [FairMOT](https://github.com/microsoft/FairMOT)

  ```
  ${DATA_DIR}
      ├── MOT15
      │   └── images
      │       ├── test
      │       └── train
      ├── MOT17
      │   └── images
      │       ├── test
      │       └── train
      └── MOT20
          └── images
              ├── test
              └── train
  ```

# Attacked Model

* We choose DLA-34: [[Google]](https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu) [[Baidu, code: 88yn]](https://pan.baidu.com/s/1YQGulGblw_hrfvwiO6MIvA) trained by [FairMOT](https://github.com/microsoft/FairMOT) as our primary attacked model.

# Tracking

* tracking on original videos of 2DMOT15, MOT17, and MOT20

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
```

# Attack

## Single-Target Attack

* attack all attackable objects separately in videos in parallel (may require a lot of memory).

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id -1
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id -1
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id -1
```

* attack a specific object in a specific video (require to set specific video in `src/track.py`).

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id ${a specific id in origial tracklets}
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id ${a specific id in origial tracklets}
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack single --attack_id ${a specific id in origial tracklets}
```

## Multiple-Targets Attack

* attack all attackable objects in videos.

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
```

# Visualization

* **attack object: 19th tracklet, target object: 24th tracklet.**

<img src="assets/original.gif" width="400"/>   <img src="assets/attacked.gif" width="400"/>

* First GIF shows the original tracklet. Second GIF shows the attacked tracklet.

