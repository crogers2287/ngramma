#!/usr/bin/env python3
"""Deterministic programmatic tasks; no tokenizer, model, teacher, or sealed data.

Run `python tasks.py --split dev` for JSONL or `--self-test` for synthetic tests.
The holdout partition is a new within-family input split, not the earlier sealed
family. Do not evaluate it while choosing rows, hints, directions, or magnitudes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import random
import unittest

SEED = 5005
FAMILIES = ('decimal_version', 'signed_arithmetic', 'state_override', 'missing_zero')


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    prompt: str
    answer: str
    hint: str
    semantic_input: tuple

    @property
    def hint_prompt(self) -> str:
        return self.hint + '\n' + self.prompt

    def to_dict(self) -> dict:
        return {**asdict(self), 'hint_prompt': self.hint_prompt}


def grade(task: Task, generated_text: str) -> dict:
    """Grade the actual complete generated text, never a constrained token score.

    Allow surrounding whitespace only. An explanation, prefix, punctuation,
    second answer, or empty result is not an accepted response. Malformed output
    is counted wrong but separately identified so it cannot masquerade as a
    demonstrated reasoning error.
    """
    if not isinstance(generated_text, str):
        raise TypeError('Pass the actual decoded generated completion as text')
    normalized = generated_text.strip()
    well_formed = normalized in ('A', 'B')
    return {'correct': well_formed and normalized == task.answer,
            'well_formed': well_formed, 'parsed_answer': normalized if well_formed else None,
            'error_kind': None if well_formed and normalized == task.answer else
                          ('wrong_choice' if well_formed else 'malformed')}


def _choice(task_id, family, split, stem, correct, distractor, hint, semantic_input, answer):
    if correct == distractor or answer not in ('A', 'B'):
        raise ValueError('Require distinct choices and a valid answer position')
    a, b = (correct, distractor) if answer == 'A' else (distractor, correct)
    return Task(task_id, family, split,
                f'{stem}\nA: {a}\nB: {b}\nAnswer only A or B.', answer, hint,
                tuple(semantic_input))


def _decimal(index, split, major, short, long, version):
    left, right = f'{major}.{short}', f'{major}.{long}'
    key = (lambda s: tuple(map(int, s.split('.')))) if version else Decimal
    correct, other = (left, right) if key(left) > key(right) else (right, left)
    mode = 'software version' if version else 'decimal number'
    hint = ('Compare version components as integers from left to right.' if version else
            'For decimal values, align decimal places; trailing zeros do not change a value.')
    return _choice(f'{split}-decimal-{index}', 'decimal_version', split,
                   f'Which {mode} is larger: {left} or {right}?', correct, other, hint,
                   (mode, left, right), 'AB'[(index + index // 2) % 2])


def _arithmetic(index, split, a, b):
    if index % 2 == 0:
        expression = f'-{a} - (-{b})'
        correct, distractor = -a - (-b), -a - b
        hint = 'Subtracting a negative adds its magnitude; then combine the signed values.'
    else:
        expression = f'(-{a}) * (-{b})'
        correct, distractor = (-a) * (-b), -(a * b)
        hint = 'Multiplying two negative numbers gives a positive result.'
    # Alternate label in pairs, so expression type does not predict the label.
    return _choice(f'{split}-arithmetic-{index}', 'signed_arithmetic', split,
                   f'Evaluate {expression}.', correct, distractor, hint,
                   (expression,), 'AB'[(index // 2) % 2])


def _state(index, split, start, first, addition, replacement, final_add):
    # Explicit sequential interpreter, including replacement rather than merge.
    operations = [('set', first), ('add', addition), ('set', replacement), ('add', final_add)]
    state = start
    for op, value in operations:
        state = value if op == 'set' else state + value
    distractor = first + addition + final_add  # Ignoring the final set.
    if distractor == state:
        raise ValueError('State distractor accidentally equals oracle')
    stem = (f'x starts at {start}. In order: set x to {first}; add {addition}; '
            f'set x to {replacement}; add {final_add}. What is x?')
    return _choice(f'{split}-state-{index}', 'state_override', split, stem,
                   state, distractor, 'A set operation replaces the current value; apply operations in order.',
                   (start, *tuple(tuple(op) for op in operations)), 'AB'[index % 2])


def _missing(index, split, default, present):
    record = {'x': 0} if present else {}
    correct = record['x'] if 'x' in record else default
    distractor = default if present else 0
    stem = (f'Use {default} only when field x is missing. Record: '
            + ('{x: 0}.' if present else '{}.') + ' What value is returned?')
    return _choice(f'{split}-missing-{index}', 'missing_zero', split, stem,
                   correct, distractor, 'A present zero is a value, not a missing field.',
                   (default, present), 'AB'[(index // 2) % 2])


def generate_tasks(split: str = 'dev') -> list[Task]:
    """Return 16 dev, 32 new-input holdout, or 8 easy regression tasks.

    Fixed seed and split-specific input domains prevent accidental resampling to
    select favorable outcomes. No external/sealed family is read.
    """
    if split == 'controls':
        return _controls()
    if split not in ('dev', 'holdout'):
        raise ValueError('split must be dev, holdout, or controls')
    count = 4 if split == 'dev' else 8
    rng = random.Random(SEED)
    tasks = []
    # Same digit-length trap, held-out major/components. Four dev examples pair
    # the same displayed inputs across decimal/version semantics deliberately.
    decimal_params = [(9,9,11), (9,9,11), (3,8,12), (3,8,12)] if split == 'dev' else [
        (major,short,long) for major,short,long in
        [(12,7,21),(12,7,21),(16,6,19),(16,6,19),
         (24,8,31),(24,8,31),(31,9,42),(31,9,42)]]
    for i, params in enumerate(decimal_params):
        tasks.append(_decimal(i, split, *params, version=bool(i % 2)))
    arithmetic_params = [(3,7),(4,6),(8,2),(5,9)] if split == 'dev' else [
        ((rng.randrange(31,50),rng.randrange(11,30)) if i % 4 == 2 else
         (rng.randrange(11,30),rng.randrange(31,50))) for i in range(count)]
    for i,(a,b) in enumerate(arithmetic_params):
        tasks.append(_arithmetic(i, split, a, b))
    for i in range(count):
        offset = 0 if split == 'dev' else 30
        tasks.append(_state(i,split,1+i+offset,4+i+offset,2+i,1+2*i+offset,3+i))
    for i in range(count):
        tasks.append(_missing(i,split,(7+i if split == 'dev' else 21+i),bool(i % 2)))
    return tasks


def _controls():
    specs = [
        ('Which decimal number is larger: 2.4 or 2.1?', '2.4','2.1'),
        ('Which software version is larger: 4.8 or 4.3?', '4.8','4.3'),
        ('Evaluate 2 + 3.',5,6),
        ('Evaluate 9 - 4.',5,4),
        ('x starts at 1. Add 2. What is x?',3,1),
        ('x starts at 9. Set x to 4. What is x?',4,9),
        ('Use 7 if x is missing. Record: {x: 3}. What value is returned?',3,7),
        ('Use 8 if x is missing. Record: {}. What value is returned?',8,0),
    ]
    return [_choice(f'controls-{i}',FAMILIES[i//2],'controls',stem,c,d,'',
                    ('control',i),'AB'[i%2]) for i,(stem,c,d) in enumerate(specs)]


class CurriculumTests(unittest.TestCase):
    def test_counts_and_balance(self):
        for split,count in [('dev',16),('holdout',32),('controls',8)]:
            tasks=generate_tasks(split)
            self.assertEqual(len(tasks),count)
            self.assertEqual(len({t.id for t in tasks}),count)
            for family in FAMILIES:
                answers=[t.answer for t in tasks if t.family==family]
                self.assertEqual(answers.count('A'),answers.count('B'))

    def test_oracle_examples(self):
        dev={t.id:t for t in generate_tasks()}
        self.assertIn('A: 9.9',dev['dev-decimal-0'].prompt)
        self.assertEqual(dev['dev-decimal-0'].answer,'A')
        self.assertIn('B: 9.11',dev['dev-decimal-1'].prompt)
        self.assertEqual(dev['dev-decimal-1'].answer,'B')
        self.assertIn('A: 4',dev['dev-arithmetic-0'].prompt)
        self.assertIn('B: -6',dev['dev-arithmetic-2'].prompt)
        self.assertIn('A: 4',dev['dev-state-0'].prompt)
        self.assertIn('A: 0',dev['dev-missing-1'].prompt)

    def test_disjoint_reproducible(self):
        dev,held=generate_tasks(),generate_tasks('holdout')
        self.assertEqual(held,generate_tasks('holdout'))
        self.assertFalse({t.semantic_input for t in dev}&{t.semantic_input for t in held})
        self.assertFalse({t.prompt for t in dev}&{t.prompt for t in held})

    def test_grader(self):
        t=generate_tasks()[0]
        self.assertTrue(grade(t,' \nA\n')['correct'])
        self.assertEqual(grade(t,'B')['error_kind'],'wrong_choice')
        for text in ('','A.','Answer: A','A because it is larger','A B','<think>x</think>A','a'):
            self.assertEqual(grade(t,text)['error_kind'],'malformed')
        with self.assertRaises(TypeError):grade(t,['A'])

    def test_short_and_hints(self):
        # Word counts are NOT model token counts; actual tokenizer must gate runs.
        for split in ('dev','holdout','controls'):
            for task in generate_tasks(split):
                self.assertLess(len(task.prompt.split()),50)
                self.assertTrue(grade(task,task.answer)['correct'])
                self.assertFalse(grade(task,'B' if task.answer=='A' else 'A')['correct'])
                if split!='controls':self.assertTrue(task.hint)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','holdout','controls'],default='dev')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CurriculumTests))
        raise SystemExit(not result.wasSuccessful())
    for task in generate_tasks(args.split):print(json.dumps(task.to_dict(),ensure_ascii=False))


if __name__=='__main__':main()
