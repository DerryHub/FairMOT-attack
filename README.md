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

