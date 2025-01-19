from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", max_seq_length=8192
)

train_ds = load_dataset("json", data_dir="./data/augmented_6/train.jsonl")
test_ds = load_dataset("json", data_dir="./data/augmented_6/test.jsonl")

print(train_ds)
print(test_ds)

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
# )
