#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=generate
#SBATCH --mail-type=END
#SBATCH --mail-user=ly571@nyu.edu
#SBATCH --output=generate_%j.out
#SBATCH --gres=gpu:1

module purge
source maskgan-env/bin/activate

HOME=/scratch/$USER
data_dir=$HOME/maskgan/tmp/ptb
base_dir=$HOME/maskgan
module purge
module load tensorflow/python2.7/1.5.0

python $base_dir/generate_samples.py \
 --data_dir=$data_dir \
 --data_set=ptb \
 --batch_size=256 \
 --sequence_length=20 \
 --base_directory=$base_dir \
 --output_path=$HOME/generated-outputs \
 --sample_mode=TRAIN \
 --output_masked_logs=False \
 --output_original_inputs=True \
 --maskgan_ckpt=$base_dir/train/model.ckpt-0 \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,gen_vd_keep_prob=0.33971" \
 --generator_model='seq2seq_vd' \
 --discriminator_model='seq2seq_vd' \
 --gen_training_strategy='reinforce' \
 --is_present_rate=0.5  \
 --dis_share_embedding=True \
 --seq2seq_share_embedding=True \
 --attention_option=luong \
 --mask_strategy=contiguous \
 --baseline_method=critic \
 --number_epochs=1

for ((i=1000;i<=200000;i+=1000)); do
	python $base_dir/generate_samples.py \
 	--data_dir=$data_dir \
 	--data_set=ptb \
 	--batch_size=256 \
 	--sequence_length=20 \
 	--base_directory=$base_dir \
 	--output_path=$HOME/generated-outputs \
 	--sample_mode=TRAIN \
 	--output_masked_logs=False \
	--output_original_inputs=False \
	--step_ckpt="$i" \
	--maskgan_ckpt=$HOME/saved-ckpts/gan-ckpt-"$i" \
 	--hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,gen_vd_keep_prob=0.33971" \
 	--generator_model='seq2seq_vd' \
 	--discriminator_model='seq2seq_vd' \
 	--is_present_rate=0.5  \
 	--dis_share_embedding=True \
	--seq2seq_share_embedding=True \
 	--attention_option=luong \
 	--mask_strategy=contiguous \
 	--baseline_method=critic \
 	--number_epochs=1;
done

deactivate
