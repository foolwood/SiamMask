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
    --config config.json \
    --dataset $dataset \
    --penalty-k 0.0,0.25,0.02 \
    --window-influence 0.36,0.51,0.02 \
    --lr 0.25,0.56,0.05 \
    --search-region 255,256,16 \
    --resume $model 2>&1 | tee logs/tune.log

