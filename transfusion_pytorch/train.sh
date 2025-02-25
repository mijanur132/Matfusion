#!/bin/bash
module load PrgEnv-gnu
module load gcc/11.2.0
module load amd-mixed/6.0.0
module load ninja
module load miniforge3

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
# export OMP_NUM_THREADS=2

export TORCH_HOME=$PWD/cache

source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh
conda activate /ccs/home/palashmr/packages/miniconda/pyt_env/transfusion


export PATH=/ccs/home/palashmr/packages/miniconda/pyt_env/transfusion/bin:$PATH
#export PATH=/lustre/orion/stf218/world-shared/palashmr/py310_frontier/bin/wandb:$PATH
#export PATH=/ccs/home/palashmr/packages/miniconda/pyt_env/score-mri-amd/bin:$PATH


export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
# module load PrgEnv-gnu
# module load gcc/11.2.0
# module load rocm/6.0.0


export ROCM_HOME=/opt/rocm-6.0.0
#export PATH=/opt/rocm-6.0.0/bin
#export ROCM_HOME=/opt/rocm-5.6.0

export NCCL_DEBUG=VERSION
export FI_CXI_ATS=0
#export LD_LIBRARY_PATH=/opt/rocm-5.6.0/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.6.0/lib:/opt/rocm-5.6.0/hip/lib
export LD_LIBRARY_PATH=/opt/rocm-6.0.0/include/rccl/build:$PWD/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-6.0.0/lib
export FI_LOG_LEVEL=info
export NCCL_NET_GDR_LEVEL=3

#export PATH="/ccs/home/palashmr/.local/crusher/miniforge3/23.11.0/bin:$PATH"

#required to solve MIOPEN error: https://github.com/pytorch/pytorch/issues/60477
#export MIOPEN_USER_DB_PATH="/tmp/cache"    #dont save in lustre... it will titmeout to read: https://github.com/pytorch/pytorch/issues/60477 
export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export TORCH_EXTENSIONS_DIR="/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/temp"  #to have a write directory
export PYTHONUNBUFFERED=1   #for immeideate printing.


# Usage function to display help
usage() {
    echo "Usage: $0 --dim_latent <int> --mod_shape <space-separated integers> --xdim <int> --xdepth <int>"
    exit 1
}

# Set default values
DIM_LATENT=56
MOD_SHAPE=28
XDIM=256
XDEPTH=4

# Parse arguments and update default values if provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dim_latent) DIM_LATENT="$2"; shift ;;
        --mod_shape) MOD_SHAPE="$2"; shift ;;
        --xdim) XDIM="$2"; shift ;;
        --xdepth) XDEPTH="$2"; shift ;;
        *) usage ;; # Unknown option
    esac
    shift
done

# Call the Python script with the arguments
srun --nodes=1 --ntasks=8 python train_transfusion.py --dim_latent $DIM_LATENT --mod_shape $MOD_SHAPE --xdim $XDIM --xdepth $XDEPTH
#srun --nodes=1 --ntasks=8 python inference_transfusion.py
#srun --nodes=1 --ntasks=8 python train_transfusion.py --dim_latent 28 --mod_shape 28 --xdim 256 --xdepth 4





