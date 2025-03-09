import json
from typing import List, Dict

def extract_json_from_text(text):
    json_start = 0
    json_end = 0
    close_brace_count = 0
    extracted_jsons = []
    for idx, char in enumerate(text):
        if char == '{':
            if close_brace_count == 0:
                json_start = idx
            close_brace_count += 1
        elif char == '}':
            close_brace_count -= 1
            if close_brace_count == 0:
                json_end = idx + 1
                extracted_json = text[json_start:json_end]
                try:
                    extracted_jsons.append(json.loads(extracted_json))
                except json.JSONDecodeError:
                    pass
    return extracted_jsons


def compute_f1(pred_list, true_list):
    tp = len(set(pred_list) & set(true_list))
    fp = len(set(pred_list) - set(true_list))
    fn = len(set(true_list) - set(pred_list))

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


def get_entities_f1_score(pred_entities: List[dict], true_entities: List[dict]) -> float:
    """
    Compute F1 score between predicted and true entities from JSON-like dictionaries.
    """
    pred_set = set()
    true_set = set()

    for pred in pred_entities:
        if isinstance(pred, dict):
            pred_set.update((key, str(value)) for key, value in pred.items())

    for true in true_entities:
        if isinstance(true, dict):
            true_set.update((key, str(value)) for key, value in true.items())

    return compute_f1(pred_set, true_set)


def json_consistency_reward(completions, answer, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        extracted_jsons = extract_json_from_text(content)
        if len(extracted_jsons) == 1:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def f1_entities_reward(completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        extracted_jsons_pred = extract_json_from_text(content)
        extracted_jsons_true = extract_json_from_text(sol)

        if len(extracted_jsons_pred) == 1 and len(extracted_jsons_true) == 1:
            # Wrap single dicts in lists for get_entities_f1_score
            f1_score = get_entities_f1_score([extracted_jsons_pred[0]], [extracted_jsons_true[0]])
            rewards.append(f1_score)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [("<think>" in content and "</think>" in content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]