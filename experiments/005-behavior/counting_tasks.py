#!/usr/bin/env python3
"""Fixed overlap-count discovery; generated holdout is within-family only.

No model, tokenizer, teacher, original sealed tasks, or file inputs. Development
cases are those fixed in DISCOVERY-EXTENSION.md before original task outcomes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import random
import re

HOLDOUT_SEED = 5051
HINT = 'Check every starting position. After a match, advance one character, not the pattern length.'
CASES = (
    ('ABABABABABABAABABABABABABA','ABA',12,6,'A'),
    ('AAAAAAAAAAAAABAAAAAAAAAAA','AAA',21,20,'B'),
    ('BAABAABAABAABBAABAABAABAAB','BAAB',4,8,'B'),
    ('ABABBAABABABAABABBAABABABA','ABA',8,9,'A'),
    ('AAAAABAAAAAABAAAAABAAAAAA','AAA',6,14,'B'),
    ('AABAABAABAABAAABAABAABAABA','AABA',8,9,'A'),
    ('BABABABABABABBABABABABABA','BAB',11,6,'A'),
    ('AAAABAAAABAAAABAAAABAAAA','AAA',11,10,'B'),
)


def matching_starts(text: str, pattern: str) -> tuple[int, ...]:
    if not isinstance(text,str) or not isinstance(pattern,str) or not pattern:
        raise ValueError('Require strings and a nonempty pattern')
    starts=tuple(i for i in range(len(text)-len(pattern)+1) if text.startswith(pattern,i))
    independent=tuple(m.start() for m in re.finditer('(?='+re.escape(pattern)+')',text))
    if starts != independent:
        raise AssertionError('Independent overlapping-match oracles disagree')
    return starts


def complement(text: str) -> str:
    return text.translate(str.maketrans('AB','BA'))


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    prompt: str
    answer: str
    hint: str
    semantic_input: tuple[str,str]
    oracle_count: int
    matching_start_indices: tuple[int,...]
    choices: tuple[int,int]

    @property
    def hint_prompt(self):
        return self.hint+'\n'+self.prompt if self.hint else self.prompt

    def to_dict(self):
        return {**asdict(self),'hint_prompt':self.hint_prompt}


def grade(task: Task, generated_text: str) -> dict:
    """Whole actual completion; whitespace only is ignored, not extra content."""
    if not isinstance(generated_text,str):
        raise TypeError('Pass the actual decoded generated completion as text')
    answer=generated_text.strip();valid=answer in ('A','B');correct=valid and answer==task.answer
    return {'correct':correct,'well_formed':valid,'parsed_answer':answer if valid else None,
            'error_kind':None if correct else ('wrong_choice' if valid else 'malformed')}


def _task(index,split,text,pattern,a,b,answer,paraphrase=False):
    starts=matching_starts(text,pattern)
    if a==b or answer not in ('A','B') or (a,b)['AB'.index(answer)]!=len(starts):
        raise ValueError('Choices do not match the programmatic oracle')
    stem=(f'How many starting positions in {text} match {pattern}, including overlaps?' if paraphrase else
          f'Count overlapping occurrences of {pattern} in {text}.')
    return Task(f'{split}-overlap-{index+1}','overlap_count',split,
                f'{stem}\nA: {a}\nB: {b}\nAnswer only A or B.',answer,
                HINT if split!='controls' else '',(text,pattern),len(starts),starts,(a,b))


def generate_tasks(split='dev') -> list[Task]:
    if split=='dev':
        return [_task(i,split,*case) for i,case in enumerate(CASES)]
    if split=='controls':
        cases=(('ABAB','AAA',0,1,'A'),('BAAB','AA',0,1,'B'),
               ('ABAABA','ABA',3,2,'B'),('AAA','AA',2,1,'A'))
        return [_task(i,split,*case) for i,case in enumerate(cases)]
    if split!='holdout':raise ValueError('split must be dev, holdout, or controls')
    rng=random.Random(HOLDOUT_SEED)
    patterns=('ABA','AAA','BAAB','AABA','BAB','BBB','ABBA')
    # Exclude whole strings and their reversals/complements, even if the pattern
    # differs. Holdout shares the family/mechanism, not an unseen-family claim.
    forbidden=set()
    def exclude(text):
        forbidden.update((text,text[::-1],complement(text),complement(text)[::-1]))
    for text,*_ in CASES:exclude(text)
    result=[]
    for i in range(16):
        require_zero=i%4==0
        for attempt in range(10000):
            text=''.join(rng.choice('AB') for _ in range(rng.randrange(24,31)))
            pattern=rng.choice(patterns);count=len(matching_starts(text,pattern))
            if text not in forbidden and ((count==0) if require_zero else (count>=2)):
                break
        else:raise RuntimeError('Fixed holdout rejection sampler exceeded bound')
        exclude(text)
        # Eight correct counts are larger, eight smaller; all distractors are
        # nonnegative. The larger-correct positions require positive counts.
        distractor=count-1 if i%2 else count+1
        answer='ABBABAAB'[i%8]
        a,b=(count,distractor) if answer=='A' else (distractor,count)
        # Last four form the fixed paraphrase subset, balanced labels and order.
        result.append(_task(i,split,text,pattern,a,b,answer,paraphrase=i>=12))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev','holdout','controls'),default='dev')
    args=parser.parse_args()
    for task in generate_tasks(args.split):print(json.dumps(task.to_dict()))


if __name__=='__main__':main()
