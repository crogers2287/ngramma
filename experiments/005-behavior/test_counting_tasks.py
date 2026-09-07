"""Model-free curriculum/oracle checks; do not evaluate holdout model behavior."""
import unittest
import counting_tasks as tasks


class CountingTests(unittest.TestCase):
    def test_exact_predeclared_cases(self):
        generated=tasks.generate_tasks()
        self.assertEqual([t.oracle_count for t in generated],[12,20,8,8,14,8,11,10])
        self.assertEqual(''.join(t.answer for t in generated),'ABBABAAB')
        self.assertEqual([t.choices for t in generated],[(12,6),(21,20),(4,8),(8,9),(6,14),(8,9),(11,6),(11,10)])
        for t,case in zip(generated,tasks.CASES):
            text,pattern,a,b,answer=case
            self.assertEqual(t.prompt,f'Count overlapping occurrences of {pattern} in {text}.\nA: {a}\nB: {b}\nAnswer only A or B.')

    def test_boundaries_and_overlap(self):
        for text,pattern,expected in [('AAAA','AA',(0,1,2)),('ABAABA','ABA',(0,3)),
                ('AAAA','B',()),('AB','ABA',()),('','A',()),('ABAB','B',(1,3)),
                ('ABAB','AB',(0,2)),('BAAAB','AA',(1,2)),('A.A.','A.',(0,2))]:
            self.assertEqual(tasks.matching_starts(text,pattern),expected)
        with self.assertRaises(ValueError):tasks.matching_starts('AAA','')

    def test_balance_and_controls(self):
        for split,size in [('dev',8),('holdout',16),('controls',4)]:
            generated=tasks.generate_tasks(split)
            self.assertEqual(len(generated),size)
            self.assertEqual(sum(t.answer=='A' for t in generated),size//2)
            self.assertEqual(len({t.id for t in generated}),size)
            self.assertTrue(all(t.choices['AB'.index(t.answer)]==t.oracle_count for t in generated))
        self.assertEqual([t.oracle_count for t in tasks.generate_tasks('controls')],[0,1,2,2])
        for split in ('dev','holdout'):
            generated=tasks.generate_tasks(split)
            larger=[t for t in generated if t.oracle_count>t.choices[1-'AB'.index(t.answer)]]
            self.assertEqual(len(larger),len(generated)//2)
            self.assertEqual(sum(t.answer=='A' for t in larger),len(larger)//2)

    def test_holdout_is_fixed_disjoint_within_family(self):
        held=tasks.generate_tasks('holdout')
        self.assertEqual(held,tasks.generate_tasks('holdout'))
        all_texts=[]
        for t in tasks.generate_tasks()+held:
            text,pattern=t.semantic_input
            self.assertNotIn(text,all_texts)
            all_texts.extend((text,text[::-1],tasks.complement(text),tasks.complement(text)[::-1]))
        self.assertEqual(sum(t.oracle_count==0 for t in held),4)
        self.assertTrue(all(24<=len(t.semantic_input[0])<=30 for t in held))
        self.assertTrue(all(3<=len(t.semantic_input[1])<=4 for t in held))
        self.assertEqual(sum(t.prompt.startswith('How many') for t in held),4)

    def test_strict_actual_completion_grader(self):
        t=tasks.generate_tasks()[0]
        self.assertTrue(tasks.grade(t,' \nA\n')['correct'])
        self.assertEqual(tasks.grade(t,'B')['error_kind'],'wrong_choice')
        for text in ('','A.','Answer: A','A B','A because','a','<think>A</think>A'):
            self.assertEqual(tasks.grade(t,text)['error_kind'],'malformed')
        with self.assertRaises(TypeError):tasks.grade(t,None)

    def test_serialization_and_prompt_bounds(self):
        import json
        for split in ('dev','holdout','controls'):
            for t in tasks.generate_tasks(split):
                self.assertEqual(json.loads(json.dumps(t.to_dict()))['answer'],t.answer)
                self.assertLess(len(t.prompt.split()),35)  # Not a tokenizer claim.
                if split!='controls':self.assertEqual(t.hint,tasks.HINT)


if __name__=='__main__':unittest.main()
