#!/bin/bash

job_work_dir=$PWD
job_result_dir=$1

if [ -z "${job_result_dir}" ]; then
  echo usage: $0 result_dir_mount
  exit -1
fi

  docker run --runtime=nvidia \
    -v $job_result_dir:/result \
    -v /mnt/resource/train:/work \
    --rm \
    --name="" \
    --shm-size=10g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --network=host \
    -t \
    -i pytorch_ml \
    bash
