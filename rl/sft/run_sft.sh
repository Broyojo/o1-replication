# run SFT on prm800k data

export WANDB_API_KEY="121d8f0d656e06f7ebd1c51e71e4dbcdb654af8e"

PROJECT_NAME=o1-replication
EXPERIMENT_NAME=qwen-2.5-1.5B-instruct-sft-test-1
MODEL=Qwen/Qwen2.5-1.5B-Instruct

CUDA_VISIBLE_DEVICES=3 torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/prm800k/train.parquet \
    data.val_files=./data/prm800k/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=8192 \
    data.train_batch_size=64 \
    data.micro_batch_size=1 \
    model.partial_pretrain=$MODEL \
    trainer.default_local_dir=../checkpoints \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb']