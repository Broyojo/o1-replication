import os

from datasets import load_dataset


def convert_prm800k_augmented(num_proc=os.cpu_count()):
    train = load_dataset(
        "json", data_files="./prm800k/data/augmented_8/train.jsonl", split="train"
    )
    test = load_dataset(
        "json", data_files="./prm800k/data/augmented_8/test.jsonl", split="train"
    )

    def process_fn(example):
        return {
            "question": example["problem"],
            "answer": "<think>"
            + "\n\n".join(example["pre_generated_steps"])
            + "</think>\n"
            + example["output"],
        }

    train = train.shuffle(seed=42).map(
        process_fn, num_proc=num_proc, remove_columns=train.column_names  # type: ignore
    )
    test = test.shuffle(seed=42).map(
        process_fn, num_proc=num_proc, remove_columns=test.column_names  # type: ignore
    )

    print(train[0])
    print(test[0])

    train.to_parquet("./data/prm800k/train.parquet")  # type: ignore
    test.to_parquet("./data/prm800k/test.parquet")  # type: ignore


def main():
    convert_prm800k_augmented()


if __name__ == "__main__":
    main()
