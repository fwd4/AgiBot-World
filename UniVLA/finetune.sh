#!/bin/bash

VLA_PATH=/share/public/lianyaoxiu/.cache/huggingface/hub/models--qwbu--univla-7b/snapshots/fccd974b2b2b7c3262e62b2a0f1b4a3a05bbdef3
#LAM_PATH=logs/task_centric_lam_stage2/last.ckpt
LAM_PATH=/share/public/lianyaoxiu/.cache/huggingface/hub/models--qwbu--univla-latent-action-model/snapshots/b5401d195f9f2ed443202ad1bbdf8c191b557242/lam-stage-2.ckpt

CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
	 scripts/finetune.py \
	 --vla_path $VLA_PATH \
	 --lam_path $LAM_PATH \
	 --data_root_dir ../data/SimData --meta_json_dir ../data/SimData \
	 --codebook_size 16 --batch_size 4 --grad_accumulation_steps 1 \
	 --max_steps 1000 --save_steps 1000 \
	 --decoder_n_layers 2 --decoder_hidden_dim 1024 \
	 --run_root_dir checkpoints/rundir --adapter_tmp_dir checkpoints/adapterdir \
	 --save_latest_checkpoint_only --with_proprio --use_lora --wogripper \
	 --task_ids 0 1 2 3 4 5 6 7 8 9
