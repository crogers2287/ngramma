"""Usage evidence for existing rows; collisions remain visible."""
from collections import defaultdict
import math

def analyze(records, table, budget=512):
    if budget not in (512,2048,16384): raise ValueError("Unsupported pilot row budget")
    stats={}
    for record in records:
        if record['partition']=='sealed_test':raise ValueError("Sealed tasks cannot select rows")
        addresses=table.global_addresses(record['tokens'])
        for position,heads in enumerate(addresses):
            for head,row in enumerate(heads):
                row=int(row)
                s=stats.setdefault(row,{'row':row,'head':head,'target_families':set(),'retention_families':set(),'occurrences':0,'observed_token_windows':set(),'gradient_cosines':[],'ablation_effects':[]})
                s['occurrences']+=1
                s['retention_families' if record.get('retention') else 'target_families'].add(record['family_id'])
                s['observed_token_windows'].add(tuple(record['tokens'][max(0,position-2):position+1]))
    ranked=[]
    for s in stats.values():
        s['target_family_count']=len(s.pop('target_families'));s['retention_family_count']=len(s.pop('retention_families'))
        s['observed_token_windows']=[list(x) for x in sorted(s['observed_token_windows'])[:32]]
        s['candidate_score']=s['target_family_count']/(1+s['retention_family_count'])
        s['selection_status']='usage_candidate_only; gradient and ablation evidence pending'
        ranked.append(s)
    ranked.sort(key=lambda x:(-x['candidate_score'],x['row']))
    # Content filters need tokenizer labels. Until those are supplied, selection is provisional.
    candidates=[s['row'] for s in ranked if s['head']>=table.heads_per_ngram and s['target_family_count']>=2][:budget]
    return {'rows':ranked,'provisional_trigram_candidates':candidates,'trainable_selection_approved':False,'missing_evidence':['gradient_direction_consistency','regression_ablation','delimiter_filter'],'observed_usage_only':True}
