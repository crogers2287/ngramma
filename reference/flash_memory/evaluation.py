"""Family-level admission. Missing checks or insufficient sample sizes reject."""
import math
from collections import defaultdict
from .artifacts import digest

def family_scores(records):
    grouped=defaultdict(list)
    seen=set()
    for record in records:
        key=(record['family_id'],record['variant_id'])
        if key in seen: raise ValueError("Duplicate evaluation variant")
        seen.add(key)
        if type(record['success']) is not bool: raise ValueError("Missing objective success result")
        grouped[record['family_id']].append(float(record['success']))
    return {k:sum(v)/len(v) for k,v in grouped.items()}

def paired_summary(baseline, candidate):
    a=family_scores(baseline);b=family_scores(candidate)
    if a.keys()!=b.keys() or not a: raise ValueError("Evaluation families do not match")
    keys_a={(x['family_id'],x['variant_id']) for x in baseline}
    keys_b={(x['family_id'],x['variant_id']) for x in candidate}
    if keys_a!=keys_b: raise ValueError("Evaluation variants do not match")
    differences=[b[k]-a[k] for k in a]
    mean=sum(differences)/len(differences)
    if len(differences)>1:
        variance=sum((x-mean)**2 for x in differences)/(len(differences)-1)
        se=math.sqrt(variance/len(differences))
    else: se=float('inf')
    return {'families':len(a),'baseline':sum(a.values())/len(a),'candidate':sum(b.values())/len(b),'gain':mean,'lower_95_normal':mean-1.96*se,'upper_95_normal':mean+1.96*se}

def admission(report, thresholds):
    reasons=[]
    for key in ('compatibility_passed','deployment_runtime_parity','independent_generation_evaluation','identity_verified','provenance_verified','sealed_test_integrity','gradient_verified','live_routing_verified'):
        if report.get(key) is not True: reasons.append('missing_or_failed:'+key)
    try:
        target=paired_summary(report['test_baseline'],report['test_candidate'])
        regression=paired_summary(report['regression_baseline'],report['regression_candidate'])
        if target['families']<thresholds['minimum_test_families']:reasons.append('insufficient_unseen_families')
        if regression['families']<thresholds['minimum_regression_families']:reasons.append('insufficient_regression_families')
        if target['gain']<thresholds['minimum_success_gain'] or target['lower_95_normal']<=0:reasons.append('unproven_unseen_improvement')
        if regression['lower_95_normal'] < -thresholds['maximum_regression_drop']:reasons.append('regression_bound_exceeded')
    except (KeyError,ValueError) as e:
        reasons.append('invalid_evaluation:'+str(e));target=regression=None
    speed=report.get('decode_speed_ratio')
    if not isinstance(speed,(float,int)) or not math.isfinite(speed) or speed<1-thresholds['maximum_relative_decode_slowdown']:reasons.append('decode_retention_unproven')
    if report.get('critical_failures',1)!=0:reasons.append('critical_regression_or_missing_check')
    return {'accepted':not reasons,'reasons':reasons,'target':target,'regression':regression,'thresholds':thresholds,'report_sha256':digest(report),'interval_note':'Paired family means; normal confidence interval is a pilot default, not an established optimal rule.'}
