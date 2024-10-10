module load amd-mixed/6.0.0
module load rocm/6.0.0

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/


export MIOPEN_USER_DB_PATH="/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

conda activate /ccs/home/palashmr/packages/miniconda/pyt_env/transfusion
