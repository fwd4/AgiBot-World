torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \
UniVLA/scripts/finetune_challenge.py \
--codebook_size 16 \
--batch_size 4 \
--grad_accumulation_steps 1 \
--max_steps 10000 \
--save_steps 1000 \
--decoder_n_layers 2 \
--decoder_hidden_dim 1024 \
--run_root_dir checkpoints/rundir \
--adapter_tmp_dir checkpoints/adapterdir \
--save_latest_checkpoint_only \
--with_proprio \
--use_lora \
--task_ids 1 \ # for task 1
# --task_ids 0 1 2 3 4 5 6 7 8 9 \ # for all 10 tasks