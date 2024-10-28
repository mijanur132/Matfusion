
#!/bin/bash
#SBATCH -A STF218
#SBATCH -J score_MRI
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err
#SBATCH -N 16
#SBATCH -t 00:59:00
#SBATCH -S 0
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -C nvme
#BATCH -q debug


source sbcast_env.sh
# module purge
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/6.0.0
module load ninja
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/

export NCCL_NET_GDR_LEVEL=3

export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`
echo "MASTER_ADDR"=$MASTER_ADDR
export NCCL_SOCKET_IFNAME=hsn
export MASTER_PORT=29500


export TORCH_EXTENSIONS_DIR="/tmp"  #to have a write directory
export PYTHONUNBUFFERED=1   #for immeideate printing.
export PYTORCH_ROCM_ARCH=gfx90a
export WANDB_PROJECT="transfusion"

N_node=$SLURM_NNODES


srun --nproc_per_node=8 train_transfusion.py




