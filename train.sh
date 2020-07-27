#!/bin/bash

(
cd ncf
./prepare_dataset.sh
)

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env \
  ncf/ncf.py \
    --checkpoint_dir $PWD \
    --data /data/cache/ml-25m \
    --threshold 0.979 2>&1 | tee -a training.log

python ncf/userinference.py model.pth /data/ml-25m/movies.csv
