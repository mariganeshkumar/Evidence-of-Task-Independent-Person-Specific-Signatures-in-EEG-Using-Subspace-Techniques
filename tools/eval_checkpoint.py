"""Evaluate a trained modified-x-vector checkpoint without retraining.

Rebuilds the model from src/library/xvector/scripts/x_vector_models.py, loads the
weights from a keras.model checkpoint, applies the same preprocessing as training
(log(x + eps), then standardisation with the training-set mean/std) and reports:
  * softmax classification accuracy on train / val / test (1-based labels, i.e.
    without the off-by-one of the training script's printout), and
  * x-vector + LDA (fit on train segment x-vectors) + cosine accuracy on val / test.
Target / non-target cosine scores are written to <out_dir>/scores.mat; the EER can
then be computed with the bundled BOSARIS toolkit, e.g. in Octave/MATLAB from src/:
  addpath(genpath('library/bosaris_toolkit')); s = load('<out_dir>/scores.mat');
  [pm, pf] = rocch(s.target_test(:)', s.non_target_test(:)'); rocch2eer(pm, pf) * 100

Usage (Python 3.6 env from requirements.txt):
  python tools/eval_checkpoint.py CHECKPOINT DATA_MAT OUT_DIR [--scripts-dir DIR] [--eps 1e-6]
    CHECKPOINT  .../dnn_models/keras.model written by the training script
    DATA_MAT    tmp/<exp_name>/subject_id_dataset_for_keras.mat written by run_ann
    OUT_DIR     directory for scores.mat (created if missing)
"""
import argparse
import os
import random
import sys

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np

np.random.seed(0)
random.seed(0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCRIPTS = os.path.join(REPO_ROOT, 'src', 'library', 'xvector', 'scripts')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('data_mat')
    ap.add_argument('out_dir')
    ap.add_argument('--scripts-dir', default=DEFAULT_SCRIPTS,
                    help='directory containing x_vector_models.py (default: %(default)s)')
    ap.add_argument('--eps', type=float, default=1e-6, help='log offset used in training (default: %(default)s)')
    args = ap.parse_args()

    import tensorflow as tf
    tf.set_random_seed(0)
    from keras import backend as K
    K.set_session(tf.Session(graph=tf.get_default_graph()))
    sys.path.insert(0, args.scripts_dir)
    from x_vector_models import get_modified_x_vector_model
    from keras.models import Model
    from keras.utils.np_utils import to_categorical
    import hdf5storage
    import scipy.io as sio
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics.pairwise import cosine_similarity

    os.makedirs(args.out_dir, exist_ok=True)
    d = hdf5storage.loadmat(args.data_mat)
    hl = d['hidden_layer_size']
    X = {k: np.log(d[k + '_data'].astype(np.float64) + args.eps) for k in ['train', 'val', 'test']}
    mu, sd = X['train'].mean(), X['train'].std()
    print('train mean %.4f std %.4f' % (mu, sd))
    X = {k: np.moveaxis((v - mu) / sd, 2, -1) for k, v in X.items()}
    Y = {k: np.squeeze(d[k + '_label']).astype(np.int64) for k in ['train', 'val', 'test']}  # 1-based
    ncls = len(np.unique(Y['train']))

    model = get_modified_x_vector_model(X['train'], to_categorical(Y['train'] - 1, ncls), X['train'].shape[1], hl)
    model.load_weights(args.checkpoint)

    for k in ['train', 'val', 'test']:
        pred = np.argmax(model.predict(X[k], batch_size=64), axis=1) + 1  # back to 1-based labels
        print('SOFTMAX %s accuracy: %.2f%% (%d/%d)' % (k, np.mean(pred == Y[k]) * 100, np.sum(pred == Y[k]), len(Y[k])))

    # x-vector + LDA + cosine, as in Train_And_Test_Mod_Xvectors_Cosine.py
    xv = Model(inputs=model.input, outputs=model.get_layer('x_vector').output)
    P = {k: xv.predict(X[k], batch_size=64) for k in X}
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(P['train'], Y['train'])
    enrol, enrol_lab = [], []
    for i in np.unique(Y['train']):
        sub = X['train'][Y['train'] == i]
        enrol.append(xv.predict(np.expand_dims(np.concatenate(list(sub), axis=1), 0)))
        enrol_lab.append(i)
    enrol = clf.transform(np.concatenate(enrol, 0))
    enrol_lab = np.array(enrol_lab)
    out = {}
    idx = np.arange(len(enrol_lab))
    for k in ['val', 'test']:
        S = cosine_similarity(clf.transform(P[k]), enrol)
        pred = enrol_lab[np.argmax(S, axis=1)]
        print('COSINE %s accuracy: %.2f%%' % (k, np.mean(pred == Y[k]) * 100))
        tgt, non = [], []
        for i in enrol_lab:
            sc = S[Y[k] == i]
            tgt.append(sc[:, i - 1])
            non.append(sc[:, idx != i - 1].flatten())
        out['target_' + k] = np.concatenate(tgt)
        out['non_target_' + k] = np.concatenate(non)
        out['pred_' + k] = pred
        out['gt_' + k] = Y[k]
    sio.savemat(os.path.join(args.out_dir, 'scores.mat'), out)
    print('saved', os.path.join(args.out_dir, 'scores.mat'))


if __name__ == '__main__':
    main()
