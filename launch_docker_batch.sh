#!/bin/bash

job_work_dir=$PWD
job_result_dir=$1
job_script=$2

chmod +x ${job_script}

if [ -z "${job_result_dir}" ] || [ -z "${job_script}" ]; then
  echo usage: $0 result_dir_mount script
  exit -1
fi

  docker run --runtime=nvidia \
    -d=true \
    -v $job_result_dir:/result \
    -v /mnt/resource/train:/work \
    -v :/work \
    --rm \
    --name="" \
    --shm-size=10g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --network=host \
    -i pytorch_ml \
    ${job_script}
