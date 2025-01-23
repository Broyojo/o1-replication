import re

from grading import grader
from process_data import extract_boxed_answer


def validate_think_format(text):
    first_check = bool(re.match(r"^<think>.*</think>.+$", text, re.DOTALL))

    open_tags = len(re.findall(r"<think>", text))
    close_tags = len(re.findall(r"</think>", text))

    return first_check and open_tags == 1 and close_tags == 1


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    1. compute format score: adhering to '<think>[reasoning]</think>[answer]' format
    2. compute correctness score: is the extracted final answer symbolically equivalent to the groundtruth final answer
    """

    # TODO: maybe print/log the arguments and reward here once in a while to take a look

    solution_str = solution_str.split("<|im_start|>assistant\n")[-1].replace(
        "<|im_end|>", ""
    )

    # with open("test3.txt", "a+") as f:
    #     f.write(
    #         solution_str + "\n" + f"groundtruth: {ground_truth}\n" + "=" * 100 + "\n"
    #     )

    # gate the correctness by the format
    if not validate_think_format(solution_str):
        return 0.0

    solution = remove_think_tags(solution_str)
    answer = extract_boxed_answer(solution)
    if answer is None:
        return 0.0
    if grader.grade_answer(answer, ground_truth):
        return 1.0
    return 0.0


if __name__ == "__main__":
    valid = "<think>reasoning process</think>final answer"
    invalid1 = "no think tags at all"
    invalid2 = "<think>only think tags</think>"
    invalid3 = "text before <think>reasoning</think>answer"
    invalid4 = "<think>reasoning</think>answer<think>reasoning</think>answer"

    print(valid, validate_think_format(valid))  # True
    print(invalid1, validate_think_format(invalid1))  # False
    print(invalid2, validate_think_format(invalid2))  # False
    print(invalid3, validate_think_format(invalid3))  # False
    print(invalid4, validate_think_format(invalid4))  # False

    sample = "<think>This is my reasoning process</think>This is the final answer"
    result = remove_think_tags(sample)
    print(result)  # Output: "This is the final answer"

    sol = "<think>let me think step by step.\nthis is my second step</think>\nIf the three numbers are in proportion to $2:4:6$, then they should also be in proportion to $1:2:3$. This implies that the three numbers can be expressed as $x$, $2x$, and $3x$. Add these values together to get: \n\\[x+2x+3x=6x=64\\]\nDivide each side by 6 and get that \n\\[x=\\frac{64}{6}=\\frac{32}{3}=10 \\frac{2}{3}\\]\nwhich is $\\boxed{10\\frac{2}{3}}$."

    print(compute_score(sol, "10\\frac{2}{3}"))
    print(compute_score(sol, "10\\frac{2}{2}"))

    print(remove_think_tags(sol))
