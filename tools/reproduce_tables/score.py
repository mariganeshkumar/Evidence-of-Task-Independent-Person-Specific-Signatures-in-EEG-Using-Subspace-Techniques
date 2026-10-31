"""Score Dataset 1 test chunks (and optionally Rini's files) with the v0.1.0 release model.

Called by tools/reproduce_tables.sh; see that script for the full workflow.

Scoring follows src/library/xvector/scripts/Train_And_Test_Mod_Xvectors_Cosine.py:
  * build the network with get_modified_x_vector_model and load the release Keras weights;
  * x-vectors from the 'x_vector' layer, projected with the release LDA (the saved
    projection; refitting the LDA gives a slightly different projection and numbers);
  * each subject is enrolled with ONE x-vector, predicted from all of its training
    chunks concatenated in time;
  * cosine similarity against the 30 enrolment vectors, argmax -> 1-based label.
The train/test matrices are read straight from the feature files in the same order as
train_and_test_eeg_biometrics.m (subject, session, task, file name), which gives arrays
identical to tmp/<exp>/subject_id_dataset_for_keras.mat, so no training run is needed.

Writes to OUT_DIR:
  scores.mat   <name>_t / <name>_n target and non-target cosine scores for tools/reproduce_tables/eer.m
               (per chunk: target = score against its own subject, non-targets = scores
               against the other 29 subjects)
  acc.json     accuracies and chunk counts
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault('PYTHONHASHSEED', '0')
import numpy as np

np.random.seed(0)
random.seed(0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO_ROOT, 'src', 'library', 'xvector', 'scripts')

# The 5 held-out Passive-Audio test recordings that also exist in Rini's files: (subject, date, time)
RECORDINGS = [('012100073', '20180819', '040327'), ('014100078', '20180822', '043400'),
              ('015100076', '20180822', '080342'), ('016100071', '20180819', '025325'),
              ('017000070', '20180822', '010640')]
TASK = 'Passive-Audio'


def subdirs(p):
    # MATLAB/Octave dir() order (byte-wise sort), directories only
    return sorted(n for n in os.listdir(p) if os.path.isdir(os.path.join(p, n)))


def load_split(feat_dir, split, subjects, sio):
    """Same traversal as train_and_test_eeg_biometrics.m. Returns X (n, ch, freq, time), 1-based labels, paths."""
    X, y, paths = [], [], []
    for i, s in enumerate(subjects):
        sd = os.path.join(feat_dir, split, s)
        if not os.path.isdir(sd):
            continue
        for ses in subdirs(sd):
            for task in subdirs(os.path.join(sd, ses)):
                d = os.path.join(sd, ses, task)
                for f in sorted(n for n in os.listdir(d) if n.endswith('.mat')):
                    X.append(sio.loadmat(os.path.join(d, f))['feature'])
                    y.append(i + 1)
                    paths.append(os.path.join(split, s, ses, task, f))
    if not X:
        sys.exit('ERROR: no feature files found under %s' % os.path.join(feat_dir, split))
    return np.moveaxis(np.stack(X).astype(np.float64), 2, -1), np.array(y), paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True, help='features/<exp>/15/1 directory (contains train/ and test/)')
    ap.add_argument('--keras-weights', required=True)
    ap.add_argument('--lda', required=True)
    ap.add_argument('--rini-features', default='', help='output of rini_features.m (omit to skip Tables 2-3)')
    ap.add_argument('--hidden-layers', default='1024,512,160', help='config.hiddenlayers (default: %(default)s)')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    import tensorflow as tf
    tf.set_random_seed(0)
    from keras import backend as K
    K.set_session(tf.Session(graph=tf.get_default_graph(),
                             config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
    sys.path.insert(0, SCRIPTS)
    from x_vector_models import get_modified_x_vector_model
    from keras.models import Model
    from keras.utils.np_utils import to_categorical
    import pickle
    import scipy.io as sio
    from sklearn.metrics.pairwise import cosine_similarity

    subjects = subdirs(os.path.join(args.features, 'train'))
    print('  loading features from %s (%d subjects)' % (args.features, len(subjects)))
    Xtr, ytr, _ = load_split(args.features, 'train', subjects, sio)
    Xte, yte, te_paths = load_split(args.features, 'test', subjects, sio)
    print('  train chunks: %d, test chunks: %d' % (len(ytr), len(yte)))
    ncls = len(subjects)
    hl = np.array([[float(v) for v in args.hidden_layers.split(',')]])

    model = get_modified_x_vector_model(Xtr, to_categorical(ytr - 1, ncls), Xtr.shape[1], hl)
    model.load_weights(args.keras_weights)
    xv = Model(inputs=model.input, outputs=model.get_layer('x_vector').output)
    with open(args.lda, 'rb') as fh:
        lda = pickle.load(fh)  # sklearn 0.19.2 LinearDiscriminantAnalysis from the release

    enr = [xv.predict(np.expand_dims(np.concatenate(list(Xtr[ytr == i]), axis=1), 0)) for i in range(1, ncls + 1)]
    enr = lda.transform(np.concatenate(enr, 0))

    def score(X):
        S = cosine_similarity(lda.transform(xv.predict(X)), enr)
        return S, np.argmax(S, axis=1) + 1

    def tn(S, y):
        """Per chunk: target = own-subject score; non-targets = scores against the other subjects."""
        t = S[np.arange(len(y)), y - 1]
        mask = np.ones_like(S, dtype=bool)
        mask[np.arange(len(y)), y - 1] = False
        return t, S[mask]

    mats, acc = {}, {}

    def add(name, S, pred, y):
        mats[name + '_t'], mats[name + '_n'] = tn(S, y)
        acc[name] = dict(acc=float(np.mean(pred == y) * 100), n=int(len(y)), correct=int(np.sum(pred == y)))

    S_full, p_full = score(Xte)
    add('full', S_full, p_full, yte)
    print('  full test set: accuracy %.2f%% (%d chunks)' % (acc['full']['acc'], len(yte)))

    if args.rini_features:
        lab = {s: i + 1 for i, s in enumerate(subjects)}
        S_u, S_r, y_u, y_r = [], [], [], []
        for s, d, t in RECORDINGS:
            if s not in lab:
                sys.exit('ERROR: subject %s not in %s/train' % (s, args.features))
            y = lab[s]
            # this recording's rows in the full test matrix: test/<subject>/<date>/Passive-Audio/<i><time>.tmp.mat
            pre = os.path.join('test', s, d, TASK) + os.sep
            idx = [k for k, p in enumerate(te_paths) if p.startswith(pre) and p.endswith(t + '.tmp.mat')]
            if not idx:
                sys.exit('ERROR: no test chunks for %s %s_%s under %s' % (s, d, t, args.features))
            Su, pu = S_full[idx], p_full[idx]
            rd = os.path.join(args.rini_features, s)
            fr = sorted(f for f in os.listdir(rd) if f.endswith(t + '.tmp.mat'))
            Xr = np.moveaxis(np.stack([sio.loadmat(os.path.join(rd, f))['feature'] for f in fr]).astype(np.float64), 2, -1)
            Sr, pr = score(Xr)
            add('user_' + s, Su, pu, np.full(len(idx), y))
            add('rini_' + s, Sr, pr, np.full(len(fr), y))
            S_u.append(Su); S_r.append(Sr); y_u.append(np.full(len(idx), y)); y_r.append(np.full(len(fr), y))
            print('  %s: %d test chunks (dataset), %d (Rini)' % (s, len(idx), len(fr)))
        for name, S, y in [('user_all', S_u, y_u), ('rini_all', S_r, y_r)]:
            S, y = np.concatenate(S), np.concatenate(y)
            add(name, S, np.argmax(S, axis=1) + 1, y)

    os.makedirs(args.out_dir, exist_ok=True)
    sio.savemat(os.path.join(args.out_dir, 'scores.mat'), {k: v.reshape(1, -1) for k, v in mats.items()})
    with open(os.path.join(args.out_dir, 'acc.json'), 'w') as fh:
        json.dump(dict(acc=acc, n_subjects=ncls, recordings=[r[0] for r in RECORDINGS] if args.rini_features else []), fh, indent=1)


if __name__ == '__main__':
    main()
