#!/usr/bin/env python3
"""Create fixed ordinary or hinted jobs, keeping all oracle fields out of prompts."""
import argparse
import json
from pathlib import Path
from tasks import generate_tasks


CHAT_TEMPLATE_SHA256 = '3a0497e618ff1b4afb12431e2e45d8edff1bbec81491ba80aaa51ff04700b1f4'


def chat(prompt):
    # Exact no-tools, one-user-message, enable_thinking=False branch of the
    # pinned artifact template. No default system message or answer prefix.
    return ('<|im_start|>user\n' + prompt.strip() + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


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
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new job path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(jobs(a.split, a.hinted, a.controls), indent=2) + '\n')


if __name__ == '__main__':
    main()
