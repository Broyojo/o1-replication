export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="121d8f0d656e06f7ebd1c51e71e4dbcdb654af8e"

PROJECT_NAME="o1-replication"
EXPERIMENT_NAME="qwen2.5-1.5-sft-PPO-test-run-1-continued" 
POLICY="./checkpoints/o1-replication/qwen2.5-1.5-sft-PPO-test-run-1/actor/global_step_50"
CRITIC="./checkpoints/o1-replication/qwen2.5-1.5-sft-PPO-test-run-1/actor/global_step_50"

# TODO: optimize params here (for multigpu, larger batch size, quicker learning, etc.); probably do a sweep
# TODO: implement GRPO trainer into veRL (PPO seems not too efficient)
# TODO: use runpod 8xH100 SMX instance: ~$10.67/hr

CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python3 train.py --config-name="ppo_trainer" \
 trainer.n_gpus_per_node=1 \
 data.train_files=./data/filtered/train.parquet \
 data.val_files=./data/filtered/test.parquet \
 data.train_batch_size=1024 \
 data.val_batch_size=1024 \
 data.max_prompt_length=1024 \
 data.max_response_length=7168 \
 actor_rollout_ref.model.path=$POLICY \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size=1 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 critic.optim.lr=1e-5 \
 critic.model.path=$CRITIC \
 critic.model.use_remove_padding=True \
 critic.model.enable_gradient_checkpointing=False \
 critic.model.fsdp_config.param_offload=False \
 critic.model.fsdp_config.grad_offload=False \
 critic.model.fsdp_config.optimizer_offload=True \
 critic.ppo_micro_batch_size=1 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME \
 +trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=1 \
 trainer.default_local_dir=./checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
