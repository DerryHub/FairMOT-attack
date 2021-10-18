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

| Video          | Attack Accuracy | Unsuccessful IDs  | Total IDs                                                    | Attacked Frames | Mean L2 Distance   | Remark |
| -------------- | --------------- | ----------------- | ------------------------------------------------------------ | --------------- | ------------------ | ------ |
| ETH-Bahnhof    | 88.0%           | 56, 52, 95        | 11, 12, 14, 15, 20, 21, 22, 26, 27, 33, 34, 35, 37, 38, 40, 41, 45, 47, 49, 52, 56, 60, 95, 97, 99 | 183             | 4.6804274945962625 |        |
| TUD-Stadtmitte | 100.0%          |                   | 1, 2, 3, 4, 6, 8                                             | 35              | 2.382598628316607  |        |
| ETH-Pedcross2  | 95.35%          | 26, 63            | 1, 2, 3, 4, 131, 6, 133, 137, 15, 16, 18, 21, 26, 30, 34, 39, 42, 43, 46, 47, 56, 58, 61, 62, 63, 64, 66, 68, 72, 77, 78, 83, 85, 91, 94, 95, 96, 100, 1<br/>09, 113, 124, 125, 127 | 626             | 4.8495303127712335 |        |
| ADL-Rundle-8   | 94.12%          | 58                | 32, 1, 3, 36, 5, 6, 58, 8, 11, 43, 14, 18, 19, 24, 26, 27, 31 | 188             | 4.2515617355387265 |        |
| ADL-Rundle-6   | 100.0%          |                   | 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 34, 35, 36, 37 | 406             | 7.414020688369356  |        |
| TUD-Stadtmitte | 100.0%          |                   | 1, 2, 3, 4, 6, 8                                             | 34              | 2.3057527156437145 |        |
| TUD-Campus     | 100.0%          |                   | 3, 5, 6, 7                                                   | 29              | 5.243588447570801  |        |
| PETS09-S2L1    | 89.47%          | 26, 27            | 1, 2, 4, 5, 6, 7, 11, 12, 14, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27 | 101             | 4.45318080647157   |        |
| ETH-Sunnyday   | /               | /                 | /                                                            | /               | /                  |        |
| KITTI-17       | 100.0%          | 2.896989020434293 | 9, 2, 13, 14                                                 | 11              | 2.896989020434293  |        |
| KITTI-13       | 100.0%          |                   | 19                                                           | 1               | 7.46950626373291   |        |
|                |                 |                   |                                                              |                 |                    |        |
|                |                 |                   |                                                              |                 |                    |        |
|                |                 |                   |                                                              |                 |                    |        |

**MOT17**

| Video | Attack Accuracy | Unsuccessful IDs | Total IDs | Attacked Frames | Mean L2 Distance | Remark |
| ----- | --------------- | ---------------- | --------- | --------------- | ---------------- | ------ |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |
|       |                 |                  |           |                 |                  |        |

