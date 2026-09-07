#!/usr/bin/env python3
"""Create fixed ordinary or hinted jobs, keeping all oracle fields out of prompts."""
import argparse
import json
from pathlib import Path
from tasks import generate_tasks


CHAT_TEMPLATE_SHA256 = '3a0497e618ff1b4afb12431e2e45d8edff1bbec81491ba80aaa51ff04700b1f4'
FORMAT_SYSTEM = 'Reply with exactly one character: A or B. Do not explain your answer.'


def chat(prompt):
    # Exact no-tools, one-user-message, enable_thinking=False branch of the
    # pinned artifact template. No default system message or answer prefix.
    return ('<|im_start|>user\n' + prompt.strip() + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def format_chat(prompt):
    return '<|im_start|>system\n' + FORMAT_SYSTEM + '<|im_end|>\n' + chat(prompt)


def followup_jobs():
    import counting_tasks
    result = jobs()  # Fresh-process repeat of the complete original dev batch.
    for item in result[2:]:
        item['id'] = 'repeat/' + item['id']
    for item in jobs(hinted=True)[2:]:
        item['id'] = 'hinted/' + item['id']
        result.append(item)
    for item in jobs(controls=True)[2:]:
        item['id'] = 'format/' + item['id']
        item['text'] = '<|im_start|>system\n' + FORMAT_SYSTEM + '<|im_end|>\n' + item['text']
        result.append(item)
    for condition in ('ordinary', 'hinted', 'controls'):
        for task in counting_tasks.generate_tasks('controls' if condition == 'controls' else 'dev'):
            prompt = task.hint_prompt if condition == 'hinted' else task.prompt
            result.append({'id': 'count-' + condition + '/' + task.id,
                           'text': format_chat(prompt),
                           'generate': {'max_new_tokens': 4, 'compact': True,
                                        'context_tokens': 128, 'top_k': 5,
                                        'score_tokens': [32, 33]}})
    return result


def state_confirmation_jobs():
    result = []
    for label, text in (('content-phrase', ' In order: set x to'),
                        ('content-hint', 'A set operation replaces the current value; apply operations in order.')):
        result.append({'id': 'tokenize/' + label, 'text': text, 'tokenize_only': True})
    for item in jobs(controls=True):
        if item.get('tokenize_only'):
            result.append(item)
            continue
        item['id'] = 'baseline/' + item['id']
        item['text'] = '<|im_start|>system\n' + FORMAT_SYSTEM + '<|im_end|>\n' + item['text']
        result.append(item)
    selected = [t for t in generate_tasks() if t.family == 'state_override']
    for condition in ('hinted', 'repeat-hinted', 'repeat-ordinary'):
        for task in selected:
            result.append({'id': condition + '/' + task.id,
                           'text': format_chat(task.prompt if condition == 'repeat-ordinary' else task.hint_prompt),
                           'generate': {'max_new_tokens': 4, 'compact': True,
                                        'context_tokens': 128, 'top_k': 5,
                                        'score_tokens': [32, 33]}})
    return result


def jobs(split='dev', hinted=False, controls=False):
    result = [{'id': 'token-A', 'text': 'A', 'tokenize_only': True},
              {'id': 'token-B', 'text': 'B', 'tokenize_only': True}]
    tasks = generate_tasks(split)
    if controls:
        tasks += generate_tasks('controls')
    for task in tasks:
        prompt = task.hint_prompt if hinted and task.split != 'controls' else task.prompt
        result.append({'id': task.id, 'text': chat(prompt),
                       'generate': {'max_new_tokens': 4, 'compact': True,
                                    'context_tokens': 128, 'top_k': 5,
                                    'score_tokens': [32, 33]}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev', 'holdout', 'controls'], default='dev')
    p.add_argument('--hinted', action='store_true')
    p.add_argument('--controls', action='store_true')
    p.add_argument('--followup', action='store_true')
    p.add_argument('--state-confirmation', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new job path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.followup and a.state_confirmation:
        raise ValueError('Select one phase')
    selected = state_confirmation_jobs() if a.state_confirmation else followup_jobs() if a.followup else jobs(a.split, a.hinted, a.controls)
    a.output.write_text(json.dumps(selected, indent=2) + '\n')


if __name__ == '__main__':
    main()
