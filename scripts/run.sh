#!/bin/sh

#$ -S /bin/sh
#$ -jc gtn-container_g1
#$ -ac d=none
#$ -cwd
#$ -j y

nvidia-smi

cd ..

. "/home/mkxie/anaconda3/etc/profile.d/conda.sh"
conda activate base 


python run.py \
  --data_name coco2014 --data_dir /home/mkxie/code/data \
  --model_name ViT-B/16 \
  --coo_derive data --sample_ratio 0.01 --coo_mode both --coo_max_n 5 --lam 0.5

# python run.py \
#   --data_name vg256 --data_dir /home/mkxie/code/data \
#   --model_name RN101 \
#   --coo_derive data --sample_ratio 0.02 --coo_mode both --coo_max_n 4 --lam 0.5
  
