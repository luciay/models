#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=maskgan
#SBATCH --mail-type=END
#SBATCH --mail-user=ad4338@nyu.edu
#SBATCH --output=slurm_maskgan.out
#SBATCH --gres=gpu:1

module purge

source maskgan-env/bin/activate
HOME=/home/$USER
data_dir=$HOME/maskgan2/maskgan/tmp/ptb
base_dir=$HOME/maskgan2/maskgan
module purge
module load tensorflow/python2.7/1.5.0

python train_mask_gan.py \
 --data_dir=$data_dir \
 --batch_size=20 \
 --sequence_length=20 \
 --base_directory=$base_dir \
 --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,dis_num_layers=2,gen_learning_rate=0.00074876,dis_learning_rate=5e-4,baseline_decay=0.99,dis_train_iterations=1,gen_learning_rate_decay=0.95" \
 --mode='TRAIN' \
 --max_steps=1 \
 --generator_model='seq2seq_vd' \
 --discriminator_model='rnn_zaremba' \
 --is_present_rate=0.5 \
 --summaries_every=1 \
 --print_every=1 \
 --max_num_to_print=3 \
 --gen_training_strategy=cross_entropy \
 --seq2seq_share_embedding \
 --gen_pretrain_steps=1000 \
 --dis_pretrain_steps=1000 \
 --dis_pretain_op=1000 \
 --gen_pretrain_learning_rate=1.0

deactivate
