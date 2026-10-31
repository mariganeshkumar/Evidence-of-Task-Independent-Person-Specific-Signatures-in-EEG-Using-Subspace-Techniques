"""Format acc.json (score.py) and eer.txt (eer.m) as the three Markdown tables."""
import json
import sys

PAPER_ACC, PAPER_EER = '76.8', '8.16'   # Paper, Table IV (modified x-vector, Dataset 1)


def main(acc_json, eer_txt, out_md):
    a = json.load(open(acc_json))
    acc, recs = a['acc'], a['recordings']
    eer = {}
    for line in open(eer_txt):
        k, v = line.split()
        eer[k] = float(v)
    f2 = lambda x: '%.2f' % x
    L = ['## Table 1: full Dataset 1 test set (%d subjects, %d test chunks)' % (a['n_subjects'], acc['full']['n']), '',
         '| Source | Acc (%) | EER (%) |', '|---|---|---|',
         '| Paper Table IV | %s | %s |' % (PAPER_ACC, PAPER_EER),
         '| Reproduced | %s | %s |' % (f2(acc['full']['acc']), f2(eer['full']))]
    if recs:
        L += ['', "## Table 2: the %d held-out recordings also in Rini's files (%d test chunks)"
              % (len(recs), acc['user_all']['n']), '',
              '| Data | Acc (%) | EER (%) |', '|---|---|---|',
              '| Your preprocessed files | %s | %s |' % (f2(acc['user_all']['acc']), f2(eer['user_all'])),
              "| Rini's unprocessed files | %s | %s |" % (f2(acc['rini_all']['acc']), f2(eer['rini_all'])),
              '', '## Table 3: per subject (test chunks: %s)'
              % ', '.join('%s=%d' % (s, acc['user_' + s]['n']) for s in recs), '',
              '| Subject | Your acc | Your EER | Rini acc | Rini EER |', '|---|---|---|---|---|']
        for s in recs:
            if acc['user_' + s]['n'] != acc['rini_' + s]['n']:
                print('WARNING: %s has %d dataset chunks but %d Rini chunks'
                      % (s, acc['user_' + s]['n'], acc['rini_' + s]['n']), file=sys.stderr)
            L.append('| %s | %s | %s | %s | %s |' % (s, f2(acc['user_' + s]['acc']), f2(eer['user_' + s]),
                                                     f2(acc['rini_' + s]['acc']), f2(eer['rini_' + s])))
    else:
        L += ['', "Tables 2-3 skipped: no Rini directory given (Rini's files are not distributed with this repository)."]
    text = '\n'.join(L) + '\n'
    open(out_md, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main(*sys.argv[1:4])
