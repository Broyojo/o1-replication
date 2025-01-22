import re

from grading import grader
from process_data import extract_boxed_answer


def validate_think_format(text):
    pattern = r"^<think>.*?</think>.+$"
    return bool(re.match(pattern, text, re.DOTALL))


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text)


def compute_score(solution_str, ground_truth) -> float:
    """
    1. compute format score: adhering to '<think>[reasoning]</think>[answer]' format
    2. compute correctness score: is the extracted final answer symbolically equivalent to the groundtruth final answer
    """

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

    print(valid, validate_think_format(valid))  # True
    print(invalid1, validate_think_format(invalid1))  # False
    print(invalid2, validate_think_format(invalid2))  # False
    print(invalid3, validate_think_format(invalid3))  # False

    sample = "<think>This is my reasoning process</think>This is the final answer"
    result = remove_think_tags(sample)
    print(result)  # Output: "This is the final answer"

    sol = "<think>\nlet me think step by step.\n</think>\nIf the three numbers are in proportion to $2:4:6$, then they should also be in proportion to $1:2:3$. This implies that the three numbers can be expressed as $x$, $2x$, and $3x$. Add these values together to get: \n\\[x+2x+3x=6x=64\\]\nDivide each side by 6 and get that \n\\[x=\\frac{64}{6}=\\frac{32}{3}=10 \\frac{2}{3}\\]\nwhich is $\\boxed{10\\frac{2}{3}}$."

    print(compute_score(sol, "10\\frac{2}{3}"))
    print(compute_score(sol, "10\\frac{2}{2}"))
