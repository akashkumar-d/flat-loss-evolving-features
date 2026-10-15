import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['primary', 'why_not', 'primary_checks', 'share', 'W', 't10', 't90', 't90_first', 'final_test', 'final_train', 'knee',
          'dAH_last', 'CE_peak_over_logp', 'v_min', 'horizon', 'rate', 'x', 'fit_strict']


def sha(fn): return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def main():
    cfn = sys.argv[1] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'campaign_C', 'analysis.json')
    ffn = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'campaign_F', 'analysis.json')
    C = json.load(open(cfn))['rows']; F = json.load(open(ffn))['rows']
    M = json.load(open(os.path.join(HERE, 'manifest.json')))
    out = {}
    for r in M['runs']:
        tw = r['paired_with']
        if r['family'] == 'D4':
            row = C[tw]; f = row['main']
            rec = dict(source='campaign C (' + row['stage'] + ')', status=row['status'],
                       pilot=dict((k, row['meta'].get(k)) for k in ('pilot_t10', 'censored', 'rate', 'x_pred', 'reused_pilot')))
        else:
            row = F[tw]; f = row['f']
            rec = dict(source=row['source'], status=row['status'], stop_reason=row.get('stop_reason'), file=row.get('file'))
        rec['endpoint'] = {k: f[k] for k in FIELDS if k in f}
        out[r['id']] = dict(twin=tw, family=r['family'], seed=r['seed'], **rec)
    doc = dict(note='Bias-free twins of the campaign-I runs, endpoint rows copied from the paper campaigns analyses.',
               sources={'campaign_C/analysis.json': sha(cfn), 'campaign_F/analysis.json': sha(ffn)}, twins=out)
    json.dump(doc, open(os.path.join(HERE, 'twins.json'), 'w'), indent=1, sort_keys=True)
    by = {}
    for v in out.values():
        b = by.setdefault(v['family'], [0, 0]); b[0] += bool(v['endpoint'].get('primary')); b[1] += 1
    print('twins.json:', len(out), 'twins; bias-free passes by family:', {k: f'{a}/{n}' for k, (a, n) in sorted(by.items())})


if __name__ == '__main__':
    main()
