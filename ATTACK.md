# ATTACK

## RUN

**origin**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6
# images are saved in {opt.output_dir}/origin
```

**single**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6 --attack single --attack_id {attack_id}
# images are saved in {opt.output_dir}/single_{attack_id}
```

**multiple**

```shell
python track.py mot --load_model ~/Disk/models/all_dla34.pth --conf_thres 0.6 --attack multiple
# images are saved in {opt.output_dir}/multiple
```

## FairMOT


