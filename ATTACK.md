# ATTACK

## RUN

**origin**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6
# images are saved in {opt.output_dir}/origin
```

**single**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6 --attack single --attack_id {attack_id} --method {method}
# images are saved in {opt.output_dir}/single_{attack_id}_{method}
# {method} is in [ids, feat, det], default ids
```

**multiple**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6 --attack multiple --method {method}
# images are saved in {opt.output_dir}/multiple_{method}
# {method} is in [ids, feat, det], default ids
```

## FairMOT

### single

#### ids

**MOT15**

| Video | Attack Accuracy | Unsuccessful IDs | Total IDs | Min Attacked Frames | Max Attacked Frames | Min Attacked Frames | Mean L2 Distance | Remark |
| ----- | --------------- | ---------------- | --------- | ------------------- | ------------------- | ------------------- | ---------------- | ------ |
|       |                 |                  |           |                     |                     |                     |                  |        |
|       |                 |                  |           |                     |                     |                     |                  |        |
|       |                 |                  |           |                     |                     |                     |                  |        |

**MOT17**

| Video | Attack Accuracy | Unsuccessful IDs | Total IDs | Min Attacked Frames | Max Attacked Frames | Min Attacked Frames | Mean L2 Distance | Remark |
| ----- | --------------- | ---------------- | --------- | ------------------- | ------------------- | ------------------- | ---------------- | ------ |
|       |                 |                  |           |                     |                     |                     |                  |        |
|       |                 |                  |           |                     |                     |                     |                  |        |
|       |                 |                  |           |                     |                     |                     |                  |        |

### multiple

#### ids

**MOT15**

| Video | Attack Accuracy | Unsuccessful IDs | Total IDs | Attacked Frames | Mean L2 Distance | Remark |
| ----- | --------------- | ---------------- | --------- | --------------- | ---------------- | ------ |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |

**MOT17**

| Video | Attack Accuracy | Unsuccessful IDs | Total IDs | Attacked Frames | Mean L2 Distance | Remark |
| ----- | --------------- | ---------------- | --------- | --------------- | ---------------- | ------ |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |

