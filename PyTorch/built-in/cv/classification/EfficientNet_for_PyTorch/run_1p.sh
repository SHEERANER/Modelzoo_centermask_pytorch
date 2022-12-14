#!/usr/bin/env bash
source env_npu.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

if [ $(uname -m) = "aarch64" ]
then
	taskset -c 0-95 python3.7 -u ${currentDir}/examples/imagenet/main.py \
    --data=/data/imagenet \
    --arch=efficientnet-b0 \
    --batch-size=512 \
    --lr=0.2 \
    --momentum=0.9 \
    --epochs=100 \
    --autoaug \
    --amp \
    --pm=O1 \
    --loss_scale=32 \
    --val_feq=10 \
    --npu=0 > ${train_log_dir}/train_1p.log 2>&1 &
else
    python3.7 -u ${currentDir}/examples/imagenet/main.py \
    --data=/data/imagenet \
    --arch=efficientnet-b0 \
    --batch-size=512 \
    --lr=0.2 \
    --momentum=0.9 \
    --epochs=100 \
    --autoaug \
    --amp \
    --pm=O1 \
    --loss_scale=32 \
    --val_feq=10 \
    --npu=0 > ${train_log_dir}/train_1p.log 2>&1 &
fi