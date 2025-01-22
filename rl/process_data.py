"""
meeting notes:
- investigate transferrability of reasoning between domains
- iterative (mix math and other things?)
- model good at reasoning and non-reasoning tasks
"""

# script for automated data pipeline
import json
import os
import random

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

"""
## Data Format:
data = {
    "data_source": data_source,
    "prompt": [
        {
            "role": "user",
            "content": question,
        }
    ],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": solution},
    "extra_info": {
        "split": split,
        "index": idx,
        "answer": answer_raw,
        "question": question_raw,
    },
}
"""

"""
Intermediate data format:

{
    problem,
    solution, # prose solution
    answer, # extracted final answer
    source # math, gsm8k, etc.
}
"""


def extract_boxed_answer(text):
    def find_matching_brace(s, start):
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == "{":
                count += 1
            elif s[i] == "}":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    boxed_answers = []
    i = 0
    while i < len(text):
        if text[i : i + 6] == "\\boxed":
            brace_start = text.find("{", i)
            if brace_start != -1:
                brace_end = find_matching_brace(text, brace_start + 1)
                if brace_end != -1:
                    answer = text[brace_start + 1 : brace_end]
                    boxed_answers.append(answer.strip())
                    i = brace_end
        i += 1

    # we don't want to deal with questions with more than 1 boxed answer
    if not boxed_answers or len(boxed_answers) > 1:
        return None

    return boxed_answers[0]


def test_answer_extractor():
    """
    Compare answer extractor to PRM800K MATH train split
    """
    with open("./data/raw/MATH/train.jsonl", "r") as f:
        problems = [json.loads(line) for line in f.readlines()]

    correct = 0
    total = 0
    for problem in tqdm(problems):
        extracted = extract_boxed_answer(problem["solution"])
        if problem["answer"] == "None":
            continue
        if extracted == problem["answer"]:
            correct += 1
        else:
            print(problem)
        total += 1

    print(f"accuracy: {correct / total * 100:.2f}%")


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def load_numina_math(num_proc=os.cpu_count()):
    # TODO: some questions seem to have characters like ①③ in them, maybe replace these with actual numbers. may not matter though because they are just used for enumerating lists and they seem to be present in the prompt
    numina_math = (
        load_dataset("AI-MO/NuminaMath-CoT", split="train")
        .filter(  # remove synthetic problems
            lambda e: e["source"]
            not in ["orca_math", "synthetic_amc", "synthetic_math"],
            num_proc=num_proc,  # type: ignore
        )
        .map(  # extract boxed answer
            lambda e: {"answer": extract_boxed_answer(e["solution"])},
            num_proc=num_proc,
        )
        .filter(  # remove questions without good answer
            lambda e: e["answer"] is not None and len(e["answer"]) > 0,
            num_proc=num_proc,
        )
        .map(  # reformat dataset
            lambda e: {
                "problem": e["problem"],
                "solution": e["solution"],
                "answer": e["answer"],
                "source": e["source"],
            },
            num_proc=num_proc,
        )
        .remove_columns(["messages"])
    )

    return numina_math


def load_harp(num_proc=os.cpu_count()):
    # TODO: automatically download this or put it on HF
    harp = (
        load_dataset(
            "json", data_files="./data/raw/HARP/HARD-Math.jsonl", split="train"
        )
        .filter(  # remove questions without good answer
            lambda e: e["answer"] is not None and len(e["answer"]) > 0,
            num_proc=num_proc,  # type: ignore
        )
        .map(  # reformat dataset
            lambda e: {
                "problem": e["problem"],
                "solution": e["solution_1"],
                "answer": e["answer"],
                "source": e["contest"] + e["year"],
            },
            num_proc=num_proc,
        )
        .remove_columns(
            [
                "year",
                "contest",
                "number",
                "level",
                "subject",
                "multiple_choice_only",
                "num_solutions",
                *(f"solution_{i}" for i in range(1, 15)),
            ]
        )
    )
    return harp


def load_harp_mcq(num_proc=os.cpu_count()):
    # TODO: automatically download this or put it on HF
    harp_mcq = (
        load_dataset(
            "json", data_files="./data/raw/HARP/HARD-Math_mcq.jsonl", split="train"
        )
        .filter(  # remove questions without good answer
            lambda e: e["answer"] is not None and len(e["answer"]) > 0,
            num_proc=num_proc,  # type: ignore
        )
        .map(  # reformat problem with mcq choices
            lambda e: {
                "problem": e["problem"]
                + "\n"
                + "\n".join(f"{k}) {v}" for k, v in e["choices"].items())
            },
            num_proc=num_proc,
        )
        .map(  # reformat dataset
            lambda e: {
                "problem": e["problem"],
                "solution": e["solution_1"],
                "answer": e[
                    "answer_choice"
                ],  # TODO: maybe have multiple answers? since MCQ letter or the value of the choice are both valid potentially
                "source": e["contest"] + e["year"],
            },
            num_proc=num_proc,
        )
        .remove_columns(
            [
                "year",
                "contest",
                "number",
                "level",
                "subject",
                "multiple_choice_only",
                "choices",
                "answer_choice",
                "num_solutions",
                *(f"solution_{i}" for i in range(1, 15)),
            ]
        )
    )

    return harp_mcq


def load_omni_math(num_proc=os.cpu_count()):
    omni_math = (
        load_dataset("KbsdJames/Omni-MATH", split="test")
        .filter(  # remove questions without good answer
            lambda e: e["answer"] is not None
            and len(e["answer"]) > 0
            and not ("\\frac" in e["answer"] and not "\\frac{" in e["answer"]),
            num_proc=num_proc,  # type: ignore
        )
        .map(  # reformat dataset
            lambda e: {
                "problem": e["problem"],
                "solution": e["solution"],
                "answer": e["answer"],
                "source": e["source"],
            },
            num_proc=num_proc,
        )
        .remove_columns(["domain", "difficulty"])
    )

    return omni_math


def load_math500(num_proc=os.cpu_count()):
    math500 = (
        load_dataset("json", data_files="./data/raw/MATH/test.jsonl", split="train")
        .map(  # reformat dataset
            lambda e: {
                "problem": e["problem"],
                "solution": e["solution"],
                "answer": e["answer"],
                "source": "math500",
            },
            num_proc=num_proc,  # type: ignore
        )
        .remove_columns(["unique_id", "level", "subject"])
    )
    return math500


def save_samples_to_jsonl(
    dataset, filename="numina_samples.jsonl", num_samples=10, random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random indices
    dataset_size = len(dataset)
    num_samples = min(num_samples, dataset_size)
    random_indices = random.sample(range(dataset_size), num_samples)

    # Take the random samples
    samples = dataset.select(random_indices)

    # Write to JSONL file
    with open(filename, "w", encoding="utf-8") as f:
        for example in samples:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved {num_samples} random samples to {filename}")


def merge_datasets():
    # TODO: maybe do fuzzy string matching to find exact duplicates
    math500 = load_math500()

    numina_math = load_numina_math()
    harp = load_harp()
    harp_mcq = load_harp_mcq()
    omni_math = load_omni_math()

    dfs = [
        numina_math.to_pandas(),  # type: ignore
        harp.to_pandas(),  # type: ignore
        harp_mcq.to_pandas(),  # type: ignore
        omni_math.to_pandas(),  # type: ignore
    ]

    merged_df = pd.concat(dfs, ignore_index=True)  # type: ignore

    deduplicated_df = merged_df.drop_duplicates(subset=["problem"], keep="first")

    math500_problems = set(math500.to_pandas()["problem"])  # type: ignore

    final_df = deduplicated_df[~deduplicated_df["problem"].isin(math500_problems)]
    final_df = final_df.reset_index(drop=True)

    final_dataset = Dataset.from_pandas(final_df, preserve_index=False)

    print(f"Initial number of examples: {len(merged_df)}")
    print(f"After deduplication: {len(deduplicated_df)}")
    print(f"After removing MATH500 questions: {len(final_dataset)}")
    print(f"Total removed: {len(merged_df) - len(final_dataset)}")

    return final_dataset


def create_verl_data(num_proc=os.cpu_count()):
    merged = merge_datasets()
    math500 = load_math500()

    def process_fn(example, idx, split):
        data = {
            "data_source": example["source"],
            "prompt": [
                {
                    "role": "system",
                    "content": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed <think> </think>, i.e., 
<think>
[reasoning process here]
</think>
[answer here].

Make sure your final answer is in \\boxed{}.""",
                },
                {"role": "user", "content": example.pop("problem")},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example.pop("answer"),
            },
            "extra_info": {"split": split, "index": idx},
        }
        return data

    train_ds = merged.map(
        process_fn,
        fn_kwargs={"split": "test"},
        with_indices=True,
        num_proc=num_proc,
        remove_columns=merged.column_names,
    ).shuffle(seed=42)
    test_ds = math500.map(
        process_fn,
        fn_kwargs={"split": "test"},
        with_indices=True,
        num_proc=num_proc,
        remove_columns=merged.column_names,
    ).shuffle(seed=42)

    train_ds.to_parquet("./data/filtered/train.parquet")
    test_ds.to_parquet("./data/filtered/test.parquet")  # type: ignore


def main():
    create_verl_data(num_proc=16)


if __name__ == "__main__":
    main()
