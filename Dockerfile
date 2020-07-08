ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

env DEBIAN_FRONTEND=noninteractive
env DEBCONF_NONINTERACTIVE_SEEN=true

run echo -e "tzdata tzdata/Areas select Etc\ntzdata tzdata/zones/Etc select UTC" > /tmp/tz
run debconf-set-selections /tmp/tz

run apt-get update &&     apt-get install libmlx5-1

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip  && pip install --no-cache-dir       mlperf-compliance       opencv-python-headless       yacs

# Instal Dllogger
RUN pip install -e git://github.com/NVIDIA/dllogger#egg=dllogger

# Create required mountpoints
RUN mkdir -p /work
RUN mkdir -p /data
RUN mkdir -p /result

# Finally configure the workspace
WORKDIR /work
ENV OMP_NUM_THREADS=1
