#!/bin/bash

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

yum install -y yum-utils
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yum-config-manager --add-repo https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo

yum install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  nvidia-container-toolkit \
  nvidia-container-runtime

sudo mkdir -p $docker_imgdir
sudo mkdir -p /etc/docker

sudo tee /etc/docker/daemon.json <<-EEOF >/dev/null
{
    "data-root": "$docker_imgdir",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EEOF

if [ -z "$(lsmod | grep nv_peer_mem)" ]; then
(
  PV="1.0-9"
  cd $(mktemp -d)
  wget "https://github.com/Mellanox/nv_peer_memory/archive/${PV}.tar.gz"
  tar xf ${PV}.tar.gz
  cd nv_peer_memory-${PV}/
  ./build_module.sh
  sudo rpmbuild --rebuild /tmp/nvidia_peer_memory-${PV}.src.rpm
  sudo rpm -ivh /root/rpmbuild/RPMS/x86_64/nvidia_peer_memory-${PV}.x86_64.rpm
)
systemctl enable nv_peer_mem
fi

gpasswd -a $admin_user docker

systemctl restart docker

