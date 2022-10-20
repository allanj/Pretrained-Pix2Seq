#!/bin/bash

set -x
set -e

git clone https://github.com/allanj/Pretrained-Pix2Seq
cd Pretrained-Pix2Seq
git checkout publaynet
mkdir publaynet
cd publaynet
$hdfs -get hdfs://haruna/user/shuweifeng/doc_und/data/publaynet/val.json
$hdfs -get hdfs://haruna/user/shuweifeng/doc_und/data/publaynet/train.json
$hdfs -get hdfs://haruna/user/shuweifeng/doc_und/data/publaynet/val.tar.gz
$hdfs -get hdfs://haruna/user/shuweifeng/doc_und/data/publaynet/train images


tar -zvxf val.tar.gz
mkdir train

cd images
#$hdfs -get /home/byte_ailab_litg/user/allan/datasets/publaynet.tar.gz
#tar -zvxf publaynet.tar.gz
#mv publaynet/train ./

for i in {10..61}
do
   echo "$i"
   tar -zvxf G_PMC${i}.tar.gz
   mv train/G_PMC${i}/* ../train/
done

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ./publaynet --pix2seq_lr \
         --large_scale_jitter --rand_target --model pix2seq --output_dir ouput_dir --batch_size 6 > publaynet.log 2>&1
