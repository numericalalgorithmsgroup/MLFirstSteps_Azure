#!/bin/bash

# User config
rg_name="NCF-Tutorial"
vm_name="NCF-Trainer"
location="southcentralus"
vm_size="Standard_NC6s_v2"
sshkey="$HOME/.ssh/ptooley.pub"
admin_user=$USER

# Azure/docker specific config
work_mount=/work
dataset_mount=/data
result_mount=/result
pytorch_image_name=pytorch_ml
deploy_workdir=/mnt/resource/deploy
docker_imgdir=/mnt/resource/docker
workdir="/mnt/resource"
bundle_files="deployment"
training_workdir=/mnt/resource/train

echo -e "Creating VM Instance\n====================\n"

echo -e "\nResource Group:"
az group create \
  --name ${rg_name} \
  --location ${location}

az vm show \
  --resource-group ${rg_name} \
  --name ${vm_name}

if [ "$?" -ne "0" ]; then
  echo -e "\nVM Instance:"
  az vm create \
    --resource-group ${rg_name} \
    --name ${vm_name} \
    --size ${vm_size} \
    --image OpenLogic:CentOS-HPC:7_7-gen2:7.7.2020042001 \
    --ssh-key-value ${sshkey} \
    --admin-username ${admin_user}
else
  echo -e "\n Using existing VM"
fi

echo -e "\nGPU Extensions:"
az vm extension set \
  --resource-group ${rg_name} \
  --vm-name ${vm_name} \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute

echo -e "\nWaiting for GPU Driver install to finish (60s)"
sleep 1m

echo -e "\n\nBuilding Docker Image\n=====================\n"

cat <<-EOF > launch_docker_interactive.sh
#!/bin/bash

job_work_dir=\$PWD
job_result_dir=\$1

if [ -z "\${job_result_dir}" ]; then
  echo usage: \$0 result_dir_mount
  exit -1
fi

  docker run --runtime=nvidia \\
    -v \$job_result_dir:$result_mount \\
    -v $training_workdir:$work_mount \\
    --rm \\
    --name="$instance_name" \\
    --shm-size=10g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --ipc=host \\
    --network=host \\
    -t \\
    -i $pytorch_image_name \\
    bash
EOF

cat <<-EOF > launch_docker_batch.sh
#!/bin/bash

job_work_dir=\$PWD
job_result_dir=\$1
job_script=\$2

chmod +x \${job_script}

if [ -z "\${job_result_dir}" ] || [ -z "\${job_script}" ]; then
  echo usage: \$0 result_dir_mount script
  exit -1
fi

  docker run --runtime=nvidia \\
    -d=true \\
    -v \$job_result_dir:$result_mount \\
    -v $training_workdir:$work_mount \\
    -v $job_work_dir:$work_mount \\
    --rm \\
    --name="$instance_name" \\
    --shm-size=10g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --ipc=host \\
    --network=host \\
    -i $pytorch_image_name \\
    \${job_script}
EOF

read -r -d '' deploy_script <<-EOF
#!/bin/bash

export rg_name=${rg_name}
export vm_name=${vm_name}
export vm_size=${vm_size}
export sshkey=${sshkey}
export admin_user=${admin_user}
export work_mount=${work_mount}
export dataset_mount=${dataset_mount}
export result_mount=${result_mount}
export pytorch_image_name=${pytorch_image_name}
export deploy_workdir=${deploy_workdir}
export docker_imgdir=${docker_imgdir}

bundled_data="$(tar cf - ${bundle_files} | gzip -9 | base64 -w0)"

setfacl -d -m 'u:$admin_user:rwX' /mnt/resource
setfacl -m 'u:$admin_user:rwX' /mnt/resource;

mkdir -p $deploy_workdir
cd $deploy_workdir

echo \$bundled_data | base64 -d | tar xvz

chmod +x deployment/*
./deployment/docker_bootstrap.sh
./deployment/build_pytorch.sh
EOF

echo -e "\nBuilding images:"
az vm extension set \
  --publisher Microsoft.Azure.Extensions \
  --version 2.0 \
  --name CustomScript \
  --resource-group $rg_name \
  --vm-name $vm_name \
  --settings "{\"script\":\"$(echo "${deploy_script}" | base64 -w0)\",\
               \"timestamp\": $(date +%s)}"

vmip=$(az vm list-ip-addresses --name ${vm_name} --query "[0].virtualMachine.network.publicIpAddresses[0].ipAddress" -o tsv)

echo -e "\nCopying training files:"
if [ -e "ncf/add_personal_ratings.py" ]; then
  echo -e "\nNOTE: Adding personal ratings to training dataset. Delete ncf/add_personal_ratings.py if this is undesirable"
fi
ssh ${vmip} mkdir -p ${training_workdir}
scp -qr ncf train.sh launch_docker_{interactive,batch}.sh ${vmip}:${training_workdir}

echo -e "\n\nRunning Training\n================\n"

echo -e "\nLaunch container:"
containerid=$(ssh ${vmip} bash -c "'cd ${training_workdir}; bash ./launch_docker_batch.sh ${training_workdir} ./train.sh'")
(ssh ${vmip} docker logs -f $containerid &)
ssh ${vmip} docker wait $containerid

echo -e "\nDownloading model data:"

scp ${vmip}:${training_workdir}/model.pth .
scp ${vmip}:${training_workdir}/predictions.csv .


echo -e "\n\nDeleting VM Instance\n====================\n"

az group delete \
  --name ${rg_name}

echo -e "\n\nDone!"


