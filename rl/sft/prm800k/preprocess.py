import json
import os

import matplotlib.pyplot as plt
from grading import grader
from tqdm import tqdm


def filter_samples(data, test=False):
    problems = []
    for item in tqdm(data, desc=f"filtering {len(data)} samples"):
        if item["is_quality_control_question"] or item["is_initial_screening_question"]:
            continue

        if item["label"]["finish_reason"] != "solution":
            continue

        if not grader.grade_answer(
            item["question"]["pre_generated_answer"],
            item["question"]["ground_truth_answer"],
        ):
            continue

        problems.append(item["question"])

    problems_dedup = {problem["problem"]: problem for problem in problems}

    if not test:
        math500_questions = set(
            sample["problem"] for sample in load_samples("./data/math500.jsonl")
        )

        for question in math500_questions:
            if question in problems_dedup:
                del problems_dedup[question]

    return list(problems_dedup.values())


def load_samples(path):
    with open(path, "r") as f:
        data = [
            json.loads(line) for line in tqdm(f.readlines(), desc=f"reading {path}")
        ]
    return data


def save_samples(path, data):
    with open(path, "w") as f:
        for item in tqdm(data, desc=f"saving to {path}"):
            f.write(json.dumps(item) + "\n")


def filter():
    correct_train = filter_samples(
        load_samples("./data/raw/phase2_train.jsonl"), test=False
    )
    correct_test = filter_samples(
        load_samples("./data/raw/phase2_test.jsonl"), test=True
    )

    print(f"len(train) = {len(correct_train)}")
    print(f"len(test) = {len(correct_test)}")

    save_samples("./data/filtered/train.jsonl", correct_train)
    save_samples("./data/filtered/test.jsonl", correct_test)


def plot_step_counts():
    samples = load_samples("./data/filtered/train.jsonl")

    print(len(samples))

    lens = [len(sample["pre_generated_steps"]) for sample in samples]
    plt.hist(lens, bins=30)
    plt.savefig("lens.png")


def main():
    filter()
    plot_step_counts()


if __name__ == "__main__":
    main()
