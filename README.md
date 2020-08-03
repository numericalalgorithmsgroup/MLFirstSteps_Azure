# A Low-Cost Introduction to Machine Learning Training on Microsoft Azure

Machine learning is becoming ever more powerful and prevalent in the modern world, and is being
used in all kinds of places from cutting-edge science to computer games, and self-driving cars to
food production.  However, it is a computationally intensive process - particularly for the initial
training stage of the model, and almost universally requires expensive GPU hardware to complete the
training in a reasonable length of time.  Because of this high hardware cost, and the increasing
availability of cloud computing many ML users, both new and experienced are migrating their
workflows to the cloud in order to reduce costs and access the latest and most powerful hardware.

This tutorial demonstrates porting an existing machine learning model to a virtual machine on the
Microsoft Azure cloud platform.  We will train a small movie recommendation model using a single
GPU to give personalised recommendations. The total cost of performing this training should be no
more than $5 using any of the single GPU instances currently available on Azure.

This is not the only way to perform ML training on Azure, for example Microsoft also offer the
[Azure ML product](https://azure.microsoft.com/en-gb/services/machine-learning/), which is designed
to allow rapid deployment of commonly used ML applications.  However, the approach we will use here
is the most flexible as it gives the user complete control over all aspects of the software
environment, and is likely to be the fastest method of porting an
existing ML workflow to Azure.


## Requirements

To follow this tutorial you will need either:

* A system with git, ssh, and the [Azure CLI] installed and logged in to a valid account. If you
  are using Windows, the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
  works well for this.

or

* An active session of the [Azure Cloud Shell]

[Azure CLI]: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
[Azure Cloud Shell]: https://shell.azure.com/


## Choosing a suitable example

Although many machine learning models require large amounts of expensive compute time to train
there are also models which can produce meaningful results from much smaller datasets using only a
few minutes of CPU or GPU time.  One such model is [Neural Collaborative Filtering] (NCF), which
can be used to produce recommendation models from data on user interaction and rating data.  This
makes it possible to work through all the steps interactively in a few minutes and for only a few
dollars in cloud costs.

[Neural Collaborative Filtering]: https://arxiv.org/abs/1708.05031 

We will be training our NCF model using the [MovieLens-25M] dataset from GroupLens.  This dataset
contains 25 million ratings of 62,000 movies from 162,000 users, along with tag genome data
characterising the genre and features of the movies in the dataset.  The resulting model can then
be used to provide recommendations of the form "if you liked movie X you will probably also like
movie Y".

[MovieLens-25M]: https://grouplens.org/datasets/movielens/

The NCF implementation used for this tutorial is taken from the [NVidia Deep Learning Examples]
repository on GitHub with a small modification to update it to use the newest MovieLens-25M dataset.

[NVidia Deep Learning Examples]: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF

[Azure CLI]: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
[Azure Cloud Shell]: https://shell.azure.com/


## GPU VM Quota on Azure

GPU-enabled VMs are publically available on Azure, however you may still need to request quota
before you can create them.  If you do not have quota for the NCv2 family of VMs then the tutorial
examples will fail to run with a message that you have exceeded quota.  

By default this tutorial assumes you will run in the **SouthCentralUS** Azure region so you should
request a quota of at least 6 vCPUs for the NCv2 family in this region.  To do this, go to the
[Azure Portal](https://portal.azure.com), go to the subscriptions area, open your subscription and
select "Usage + Quotas" from the sidebar.  From this pane you can request additional quota using
the "Request Increase" button.  For a detailed guide see the
[Microsoft Documentation page](https://docs.microsoft.com/en-us/azure/azure-portal/supportability/regional-quota-requests).

## Quickstart: Scripted VM Setup and Training

The entirety of the training process can be scripted using the Azure CLI and standard Linux tools.
An example script for doing this is provided as `deploy_and_run_training.sh`

This script executes all the commands shown below to create the VM instance and run the training.
The [custom-script] VM extension is used to manage the installation of the docker and building of
the image.

[custom-script]: https://docs.microsoft.com/en-us/azure/virtual-machines/extensions/custom-script-linux

To run the example, first ensure that you are logged into the Azure CLI, then you will need to edit
the `deploy_and_run_training.sh` script to provide your personal ssh key - this is needed to allow
the script to log into the training VM. You can then run the script which will set up the VM
instance, run the training, download results and delete VM instance. When the script completes you
should have the final trained weights final predictions for one of the users, and a training log
downloaded to files named `model.pth` `predictions.csv` and `training.log`, respectively, in your
working directory.

**Note: The script will attempt to clean up all resources after use, but it is strongly recommended
to check this manually in the Azure portal to avoid a nasty - and expensive - surprise if something
goes wrong.**


## Setting Up a Training Instance

The NCF model with the MovieLens dataset is small enough to be trained in just a few minutes on a
single GPU (P100 or V100), so to begin with we will set up a single VM instance and deploy a docker
container with PyTorch that we can use to train the model.  The instance type we used was
"Standard_NC6s_v2", which contains a single NVidia P100, however you can use any instance type you
like so long as it has an NVidia P100 or V100 - only the training time of the model should change.

**All of the setup commands below are contained in the `deploy_and_run_training.sh` script - see
the "Scripting the VM Setup and Training" section below**

First we will create a new resource group to hold the VM and its related materials.  This allows
easy management and deletion of the resources we have used when they are no longer needed.

Throughout this tutorial values inside angle brackets (`<>`) represent user-specific choices of
names and options and should be replaced with an appropriate value when executing the command:

```shell
$ az group create --name <rg_name> --location SouthCentralUS
```
**Note: you can use another region, but make sure it is one where NCv2 VMs are available.** (Use
https://azure.microsoft.com/en-us/global-infrastructure/services/ to check availability.)

Then create a VM instance in this resource group:

```shell
$ az vm create \
  --resource-group <rg_name> \
  --name <vm_name> \
  --size Standard_NC6s_v2 \
  --image OpenLogic:CentOS-HPC:7_7-gen2:7.7.2020042001 \
  --ssh-key-value <sshkey> \
  --admin-username <admin_user>
```

Then the GPU driver extension needs to be installed:

```shell
$ az vm extension set \
  --resource-group <rg_name> \
  --vm-name <vm_name> \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute 
```

**Note: Currently this extension continues to perform actions after it reports being completed. You
may need to wait up to an additional 10 minutes for the instance to install additional packages and
reboot before the next steps can be done.**

After this completes connect to the instance using ssh. To find the public IP address for the
instance use:

```shell
$ az vm list-ip-addresses --name <vm_name>
```


## Getting a copy of the tutorial repository

One we are logged into the VM instance, we need to acquire a local copy of the tutorial repository.
We will do this on the local ssd of the instance, which is mounted at `/mnt/resource`:

```shell
$ sudo mkdir /mnt/resource/work
$ sudo chown -R $USER /mnt/resource/work
$ cd /mnt/resource/work
$ git clone https://github.com/numericalalgorithmsgroup/MLFirstSteps_Azure
$ cd MLFirstSteps_Azure
```

The `MLFirstSteps_Azure` directory contains all the materials needed to complete this tutorial.  The ncf
model and training scripts are located in the `ncf` subdirectory of this repository.  This
directory will be mounted in the docker container in the next step.


## Installing the Docker and Building the Image

Once we have logged into the instance, we need to install the docker with the NVidia runtime. Since
we are using the CentOS image on our VM, we can use use `yum` as shown:

```shell
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

$ sudo yum install -y yum-utils

$ sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
$ sudo yum-config-manager --add-repo https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo

$ sudo yum install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  nvidia-container-toolkit \
  nvidia-container-runtime
```

**Important Note: You may receive a message that yum is "waiting for a lock".  This can occur when
azure extensions are still running in the background**

It is also necessary to instruct the docker to use a different directory to store container data as
there is insufficient free space on Azure VM OS images:

```shell
$ sudo mkdir -p /mnt/resource/docker
$ sudo mkdir -p /etc/docker

$ sudo tee /etc/docker/daemon.json <<-EOF >/dev/null
{
    "data-root": "/mnt/resource/docker",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
```

Finally, add your username to the docker group and restart the docker service:

```shell
$ sudo gpasswd -a $USER docker

$ sudo systemctl restart docker
```

Now we need to exit the ssh session and log back in for the user permission changes to take
effect.

Once we have logged back we can build the image using the Dockerfile provided in the repository:

```shell
$ cd /mnt/resource/work/MLFirstSteps_Azure
$ docker build --rm -t pytorch_docker . -f Dockerfile
```

## Running the Training

To run the training, first it is necessary to launch the docker container, mounting the training
scripts directory as `/work` within the container.

```shell
$ cd /mnt/resource/work/MLFirstSteps_Azure
$ docker run --runtime=nvidia \
    -v /mnt/resource/work/MLFirstSteps_Azure/:/work \
    --rm \
    --name="container_name" \
    --shm-size=10g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --network=host \
    -t \
    -i pytorch_docker \
    bash
```

The final step before running the training is to download and prepare the dataset.  This is done
using the `prepare_dataset.sh` script:

```shell
$ cd /work/ncf
$ ./prepare_dataset.sh
```

Finally, run the training. The DeepLearningExamples repository [readme] gives details of the
various options that can be passed to the training.  For this example, we will run the training
until accuracy of `0.979` is attained.

[readme]: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF

```shell
$ python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env ncf.py \
  --data /data/cache/ml-25m \
  --checkpoint_dir /work \
  --threshold 0.979
```
## Inferencing: Recommending Movies

Having trained the model we can now use it to recommend additional movies.  For each user/movie
pairing the model gives a predicted user rating between 0-1. The highest predicted movies not rated
by the user can then be used as recommendations for that user.

The provided `ncf/userinference.py` script gives an example of how to generate predictions from the
trained model. It can be run either on the remote machine or on a local machine with PyTorch
installed and does not require a GPU to run. It takes two command-line arguments, the first the
path to the trained model file, and the second the path to the original `movies.csv` file from the
dataset - this is used to map movie IDs back to their names.

```shell
$ python userinference.py /work/model.pth /data/ml-25m/movies.csv --output-dir /work
```

The script will output the predictions sorted by descending rating in the file `predictions.csv`.

By default, the script will generate a predicted rating for all movies in the dataset for the
highest user ID number.

## Dowloading the results to your local machine

The results can now be copied back via ssh secure copy. To do this, use scp from your _local
machine_:

```shell
$ scp <vm_ip>:/mnt/resource/work/MLFirstSteps_Azure/model.pth .
$ scp <vm_ip>:/mnt/resource/work/MLFirstSteps_Azure/first_steps_example/predictions.csv . 
```

## Deleting the Instance After Use

To avoid being billed for more resources than needed it is important to delete the VM instance and
associated resources after use.

Ideally, if you created a resource group specifically for the tutorial resources, the whole group
can be deleted at once:

```shell
$ az group delete --name <rg_name>
```

Alternatively, if you wish to retain the other resources and delete just the VM instance, use the
`az vm delete` command:

```shell
$ az vm delete --resource-group <rg_name> --name <vm_name>
```

**Either way, make sure that you've deleted everything you expected to by looking in the Azure
Portal**

## Conclusions and Additional Resources

Having followed this tutorial you should have an idea of the steps involved in deploying an
existing machine learning workflow on the Azure platform using Docker containers. The key steps in
a such a workflow are:

1. Creating a suitable virtual machine instance and connecting to it via SSH
2. Installing the machine learning framework - for example using Docker
3. Preparing the model and data
4. Running the training
5. Downloading the results
6. Cleaning up resources after use

Now you have got started training a simple ML model, the next step is to use distributed training
with multiple GPUs and multiple VMs to train large models.  To learn how to do this on MS Azure,
check out our second ML tutorial "[Distributed Mask R-CNN Training on MS Azure]"

[Distributed Mask R-CNN Training on MS Azure]: https://github.com/numericalalgorithmsgroup/MaskR-CNN_Azure

## What Haven't We Told You (Yet)?

Running in the cloud gives you a lot of flexibility in terms of the machine types and pricing
options available. Here are a couple of extra things to consider if you are planning to deploy ML
to the cloud in earnest:

### Spot Pricing

By default Azure VMs are charged at "Pay as you go" pricing rates.  This gives you guaranteed
access to the VM until you choose to stop it, but it is the most expensive way to pay for compute
on Azure.

If you have a workload that can be interrupted, or are happy to take the risk that your job might
not complete, you can use the [spot pricing tier] instead.  This allows you to purchase unused
compute capacity at a large discount (typically 80-90%) with the risk that your workload could be
evicted at any time.

[spot pricing tier]: https://azure.microsoft.com/en-gb/pricing/spot/

To make use of the spot pricing tier you can pass the `--priority Spot` option to the Azure cli
when creating your VM.

### Different instance types

The Azure platform offers a variety of different GPU instances with different types and numbers of
GPU available.  In this tutorial we have used the "Standard\_NC6s\_v2" VM type with a single NVidia
P100, however we could also run this training on a V100 equipped VM or a VM with multiple GPUs.  To
do this we would simply change the requested VM size when calling `az vm create`.  For example, for
faster result at higher cost than the Standard\_NC6s\_v2, we could use a Standard\_NC12s\_v2
instance which is equipped with 2 P100 GPUs.  Then to take advantage of both GPUs when training
pass a value of 2 to `--nproc-per-node` when launching the training.

## NAG Cloud Computing Solutions

NAG offer [cloud computing consultancy] services to help organisations optimise their numerical
applications for the cloud. For impartial, vendor agnostic advice on high performance computing and
to find out how NAG can help you migrate to the cloud contact us at
[info@nag.co.uk](info@nag.co.uk).

[cloud computing consultancy]: https://www.nag.co.uk/content/cloud-computing-solutions
