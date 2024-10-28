ENV_NAME="transfusion"
ENV_PATH="/lustre/orion/bif151/world-shared/palashmr"

TAR_FILE=${ENV_NAME}.tar.bz2
NNODES=${SLURM_NNODES}
echo "broadcasting ${TAR_FILE}"
TIMEFORMAT="%Rs"
{ time sbcast -pf /lustre/orion/bif151/world-shared/palashmr/${TAR_FILE} /mnt/bb/${USER}/${TAR_FILE}; } 2>&1
if [ ! "$?" == "0" ]; then
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes
    echo "SBCAST failed!"
    exit 1

fi
echo "creating local directory"
srun -N ${NNODES} --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${ENV_NAME}
echo "activating global environment"
source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh

#conda activate ${ENV_NAME} # needed for lbzip2
echo "Environment: ${ENV_PATH}/${ENV_NAME}"
conda activate "${ENV_PATH}/${ENV_NAME}"
echo "untaring"
{ time srun -N ${NNODES} --ntasks-per-node 1 --cpus-per-task=64 tar -I lbzip2 -xf /mnt/bb/${USER}/${TAR_FILE} -C /mnt/bb/${USER}/${ENV_NAME}; } 2>&1
echo "activating local environment"
conda activate /mnt/bb/${USER}/${ENV_NAME}
echo "unpacking"
{ time srun -N ${NNODES} --ntasks-per-node 1 conda-unpack; } 2>&1