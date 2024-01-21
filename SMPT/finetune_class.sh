#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -G, --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH -o slurm_fc_tr.log
root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH

while getopts "d:s:l:r:" arg
do
	case $arg in
		d) dataset_name="$OPTARG";;
		s) split_type="$OPTARG";;
		l) encoder_lr="$OPTARG";
		    head_lr="$OPTARG";;
		r) dropout_rate="$OPTARG";;
	esac
done


# data, train
task="train"



# pretrain paramter
subgraph_archi="ab_ba_da"  # ab, ab_ba, ab_ba_da
pretrain_para="bs512_lr0.001_dr0.2"
pretrain_epoch="99"  # 99
init_model="pretrain_models/zinc_[$subgraph_archi]_$pretrain_para/epoch$pretrain_epoch.pdparams"

batch_size=32


log_dir="pe${pretrain_epoch}_bs${batch_size}_el${encoder_lr}_hl${head_lr}_dr${dropout_rate}_st${split_type}"
python -m paddle.distributed.launch --log_dir ./log/class/$dataset_name/$subgraph_archi/$log_dir \
  finetune_class.py \
  --task=$task \
  --dataset_name=$dataset_name \
  --split_type=$split_type \
  --subgraph_archi=$subgraph_archi \
  --init_model=$init_model \
  --batch_size=$batch_size \
  --num_workers=8 \
  --max_epoch=100 \
  --encoder_lr=$encoder_lr \
  --head_lr=$head_lr \
  --dropout_rate=$dropout_rate \
