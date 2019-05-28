if [ -z "$3" ]
  then
    echo "Need input parameter!"
    echo "Usage: bash `basename "$0"` \$MODEL \$DATASET \$GPUID"
    exit
fi

ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

model=$1
dataset=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python -u $ROOT/tools/test.py \
    --config config.json \
    --resume $model \
    --mask \
    --dataset $dataset 2>&1 | tee logs/test.log

