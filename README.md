# TraSw for FairMOT

* **A Single-Target Attack example (Attack ID: 19; Screener ID: 24):**

<table>
    <tr>
        <td ><center><img src="assets/original.gif" ><b> Fig.1  Original </b> </center></td>
        <td ><center><img src="assets/attacked.gif" ><b> Fig.2  Attacked </b> </center></td>
    </tr>
</table>
By perturbing only two frames in this example video, we can exchange the 19th ID and the 24th ID completely. Starting from frame 592, the 19th and 24th IDs can keep the exchange without noise.


> [**TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**](https://arxiv.org/abs/2111.08954),            
> Delv Lin, Qi Chen, Chengyu Zhou, Kun He,              
> *[arXiv 2111.08954](https://arxiv.org/abs/2111.08954)*

**Related Works**

* [TraSw for ByteTrack](https://github.com/DerryHub/ByteTrack-attack)

## Abstract

Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack.

## Attack Performance

**Single-Target Attack Results on MOT challenge test set**

| Dataset | Suc. Rate | Avg. Frames | Avg.  L<sub>2</sub> Distance |
| :-----: | :-------: | :---------: | :--------------------------: |
| 2DMOT15 |  95.37%   |    4.67     |             3.55             |
|  MOT17  |  96.35%   |    5.61     |             3.23             |
|  MOT20  |  98.89%   |    4.12     |             3.12             |

**Multiple-Target Attack Results on MOT challenge test set**

| Dataset | Suc. Rate | Avg.  Frames (Proportion) | Avg. L<sub>2</sub> Distance |
| :-----: | :-------: | :-----------------------: | :-------------------------: |
| 2DMOT15 |  81.95%   |          35.06%           |            2.79             |
|  MOT17  |  82.01%   |          38.85%           |            2.71             |
|  MOT20  |  82.02%   |          54.35%           |            3.28             |

## Installation

* **same as** [FairMOT](https://github.com/microsoft/FairMOT)

* Clone this repo, and we'll call the directory that you cloned as ${FA_ROOT}

* Install dependencies. We use python 3.7 and pytorch >= 1.2.0

* ```shell
  conda create -n FA
  conda activate FA
  conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
  cd ${FA_ROOT}
  pip install -r requirements.txt
  cd src/lib/models/networks/DCNv2 sh make.sh
  ```

* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo.

* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

* We only use the same test data as [FairMOT](https://github.com/microsoft/FairMOT).

* 2DMOT15, MOT17 and MOT20 can be downloaded from the official webpage of [MOT-Challenge](https://motchallenge.net/). After downloading, you should prepare the data in the following structure:

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

## Target Model

* We choose DLA-34: [[Google]](https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu) [[Baidu, code: 88yn]](https://pan.baidu.com/s/1YQGulGblw_hrfvwiO6MIvA) trained by [FairMOT](https://github.com/microsoft/FairMOT) as our primary target model.

## Tracking without Attack

* tracking on original videos of 2DMOT15, MOT17, and MOT20

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
```

## Attack

### Single-Target Attack

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

### Multiple-Targets Attack

* attack all attackable objects in videos.

```shell
cd src
python track.py mot --test_mot15 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
python track.py mot --test_mot17 True --load_model all_dla34.pth --conf_thres 0.4 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
python track.py mot --test_mot20 True --load_model all_dla34.pth --conf_thres 0.3 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --attack multiple
```

## Acknowledgement

This source code is based on [FairMOT](https://github.com/microsoft/FairMOT). Thanks for their wonderful works.

## Citation

```
@misc{lin2021trasw,
      title={TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking}, 
      author={Delv Lin and Qi Chen and Chengyu Zhou and Kun He},
      year={2021},
      eprint={2111.08954},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

