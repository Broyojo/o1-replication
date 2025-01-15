import json
import os
import random

from openai import OpenAI
from preprocess import load_samples
from tqdm import tqdm
from transformers import AutoTokenizer


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
    prompt = f"""The following is a user asking a question to an assistant which thinks internally before providing its solution to the user:\n\n{format_fake_template(sample)}\n\nPlease create an assistant output which summarizes the solution and reasoning process. The user only sees the output from the assistant, not the intermediate thinking process, so make sure that the assistant output is fluent and complete. ONLY output the text that should go into the {{assistant output}} section, nothing else."""

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


client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)


def generate_completion_for_sample(
    sample, model: str, temperature: float = 0.8, top_p=0.95
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


def main():
    train = load_samples("./data/filtered/train.jsonl")
    test = load_samples("./data/filtered/test.jsonl")

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

    # print(
    #     "approximate tokens for train:",
    #     estimate_tokens(train, tokenizer, step_to_output_tokens_constant=400),
    # )
    # print(
    #     "approximate tokens for test:",
    #     estimate_tokens(test, tokenizer, step_to_output_tokens_constant=400),
    # )

    samples = random.sample(train, k=10)

    for sample in tqdm(samples, desc="generating completion"):
        completion = generate_completion_for_sample(
            sample, model="deepseek-ai/DeepSeek-V3"
        )
        print(completion)
        assert completion

        with open(f"outputs/output_{len(os.listdir("outputs"))}.txt", "w") as f:
            f.write(
                format_fake_template(sample).replace("{assistant output}", completion),
            )


if __name__ == "__main__":
    main()
