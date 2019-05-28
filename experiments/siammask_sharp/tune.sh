if [ -z "$1" ]
  then
    echo "Need input parameter!"
    echo "Usage: bash `basename "$0"` \$MODEL \$DATASETi \$GPUID"
    exit
fi

which python

ROOT=`git rev-parse --show-toplevel`
source activate siammask
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

model=$1
dataset=$2
id=$3

CUDA_VISIBLE_DEVICES=$id python -u $ROOT/tools/tune_vot.py\
    --config config_vot18.json \
    --dataset $dataset \
    --penalty-k 0.08,0.13,0.01 \
    --window-influence 0.38,0.44,0.01 \
    --lr 0.3,0.35,0.01 \
    --search-region 255,256,16 \
    --mask --refine \
    --resume $model 2>&1 | tee logs/tune.log

