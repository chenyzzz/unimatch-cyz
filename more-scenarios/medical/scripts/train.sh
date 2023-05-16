#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['acdc']
# method: ['unimatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1', '3', '7']
dataset='acdc'
method='unimatch'
exp='unet'
split='1'

config=configs/acdc.yaml
labeled_id_path=splits/acdc/1/labeled.txt
unlabeled_id_path=splits/acdc/1/unlabeled.txt
save_path=exp/acdc-save/unimatch/unet/split-1


mkdir -p save_path

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=1234 \
    unimatch.py \
    --config=configs/acdc.yaml --labeled-id-path=splits/acdc/1/labeled.txt --unlabeled-id-path=splits/acdc/1/unlabeled.txt \
    --save-path=exp/acdc-save/unimatch/unet/split-1 --port=1234 2>&1 | tee exp/acdc-save/unimatch/unet/split-1/$now.log
#    --save-path=exp/acdc-save/unimatch/unet/split-1 --port=1234 2>&1 | tee $save_path/$now.log
