import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from preprocess import load_samples, save_samples
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def format_sample_steps(steps):
    step_prompt = ""
    for step in steps:
        step_prompt += f"<step>{step}</step>\n"
    final = f"<assistant>\n<thinking>\n{step_prompt}</thinking>\n\n{{assistant output}}\n</assistant>"
    return final


def format_fake_template(sample):
    question = sample["problem"]
    thinking = format_sample_steps(sample["pre_generated_steps"])
    prompt = f"""<user>{question}</user>\n\n{thinking}"""
    return prompt


def format_user_prompt(sample):
    prompt = f"""The following is a user asking a question to an assistant which thinks internally before providing its solution to the user:\n\n{format_fake_template(sample)}\n\nPlease create an assistant output which summarizes the solution and reasoning process. The user only sees the output from the assistant, not the intermediate thinking process, so make sure that the assistant output is fluent, complete, and doesn't include irrelevant parts from the reasoning process, such as mistakes and their corrections, extensive trial and error, rambling, scratch work, etc. Use the present tense, refrain from using 'I' in the output, and make sure to present a detailed solution to the problem, not a thought-by-thought retelling of the reasoning process. ONLY output the text that should go into the {{assistant output}} section, nothing else."""

    return prompt


def format_sample_for_output_generation(sample):
    system_prompt = "You are a helpful assistant which follows the user's instructions."
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": format_user_prompt(sample),
        },
    ]
    return messages


def estimate_tokens(samples, tokenizer, step_to_output_tokens_constant=400):
    total_tokens = 0
    for sample in tqdm(samples, desc=f"counting tokens for {len(samples)} samples"):
        prompt = format_sample_for_output_generation(sample)
        num_tokens = len(tokenizer.apply_chat_template(prompt))
        total_tokens += (
            num_tokens
            + len(sample["pre_generated_steps"]) * step_to_output_tokens_constant
        )
    return total_tokens


def generate_completion_for_sample_openai(
    sample,
    client: OpenAI,
    model: str,
    temperature: float = 0.8,
    top_p=0.95,
):
    messages = format_sample_for_output_generation(sample)
    print(json.dumps(messages, indent=4))
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


def run_vllm_generation(
    samples: list[dict],
    model: str,
    temperature: float = 0.8,
    top_p=0.95,
    max_tokens=8192,
    gpu_memory_utilization=0.9,
    max_model_len=8192 + 2048,
):
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = []
    for sample in samples:
        prompts.append(format_sample_for_output_generation(sample))

    outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
    completions = []
    for output in outputs:
        generated_text = output.outputs[0].text
        completions.append(generated_text)
    return completions


def run_openai_generation(samples, model="deepseek-ai/DeepSeek-V3", together=True):
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY") if together else None,
        base_url="https://api.together.xyz/v1" if together else None,
    )

    for sample in tqdm(samples, desc="generating completion"):
        completion = generate_completion_for_sample_openai(sample, client, model)
        print(completion)
        assert completion

        with open(f"outputs/output_{len(os.listdir("outputs"))}.txt", "w") as f:
            f.write(
                format_fake_template(sample).replace("{assistant output}", completion),
            )


def augment_samples(samples, model):
    completions = run_vllm_generation(samples, model=model)
    for sample, completion in zip(samples, completions):
        sample["output"] = completion


def compare_augmented_samples(path1, path2, model1, model2):
    samples1 = load_samples(path1)
    samples2 = load_samples(path2)

    assert len(samples1) == len(samples2) and all(
        s1["problem"] == s2["problem"] for s1, s2 in zip(samples1, samples2)
    ), "different samples"

    tokenizer1 = AutoTokenizer.from_pretrained(model1)
    tokenizer2 = AutoTokenizer.from_pretrained(model2)

    tokens1 = []
    tokens2 = []

    # compare samples on a per question basis as well as in aggregate
    for sample1, sample2 in tqdm(
        zip(samples1, samples2),
        desc=f"computing tokens for {model1} and {model2}",
        total=min(len(samples1), len(samples2)),
    ):
        num_tokens_1 = len(tokenizer1.tokenize(sample1["output"]))
        num_tokens_2 = len(tokenizer2.tokenize(sample2["output"]))

        tokens1.append(num_tokens_1)
        tokens2.append(num_tokens_2)

    # plt.figure(figsize=(10, 6))
    # plt.hist(tokens1, alpha=0.5, color="blue", bins=30, label=model1)
    # plt.hist(tokens2, alpha=0.5, color="orange", bins=30, label=model2)
    # plt.xlabel("Number of tokens")
    # plt.ylabel("Frequency")
    # plt.title("Token Distribution Comparison")
    # plt.legend()
    # plt.savefig("comparison.png")
    # plt.close()

    plt.figure(figsize=(10, 6))
    bins = np.linspace(
        min(min(tokens1), min(tokens2)), max(max(tokens1), max(tokens2)), 30
    )
    plt.hist(
        [tokens1, tokens2],
        bins=bins,
        label=[path1, path2],
        color=["blue", "orange"],
        alpha=0.7,
        rwidth=0.8,
    )
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.title("Token Distribution Comparison")
    plt.yscale("log")
    plt.legend()
    plt.savefig("comparison.png")
    plt.close()


def create_mock_train_samples(samples, path="mock.txt", k=10):
    samples = random.sample(samples, k=k)

    with open(path, "w") as f:
        for sample in samples:
            template = format_fake_template(sample)
            template = template.replace("{assistant output}", sample["output"])
            f.write(template + "\n" + "=" * 100 + "\n")


def main():
    # train = load_samples("./data/filtered/train.jsonl")
    # test = load_samples("./data/filtered/test.jsonl")

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

    # print(
    #     "approximate tokens for train:",
    #     estimate_tokens(train, tokenizer, step_to_output_tokens_constant=400),
    # )
    # print(
    #     "approximate tokens for test:",
    #     estimate_tokens(test, tokenizer, step_to_output_tokens_constant=400),
    # )

    # model = "Qwen/Qwen2.5-32B-Instruct"

    # augment_samples(test, model=model)
    # save_samples("./data/augmented_6/test.jsonl", test)

    # augment_samples(train, model=model)
    # save_samples("./data/augmented_6/train.jsonl", train)

    # compare_augmented_samples(
    #     path1="./data/augmented/test.jsonl",
    #     path2="./data/augmented_3/test.jsonl",
    #     model1="meta-llama/Llama-3.1-8B-Instruct",
    #     model2="Qwen/Qwen2.5-32B-Instruct",
    # )

    create_mock_train_samples(load_samples("./data/augmented_6/train.jsonl"), k=100)


if __name__ == "__main__":
    main()
