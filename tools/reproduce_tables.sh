#!/usr/bin/env bash
# Reproduce the three results tables with the pretrained model from GitHub release v0.1.0.
#
#   Table 1  full Dataset 1 test set: accuracy and EER (vs. the paper's Table IV)
#   Table 2  the 5 held-out recordings that also exist in Rini's files (unprocessed Net
#            Station exports of the same sessions): dataset files vs. Rini's files
#   Table 3  the same, per subject
#
# No training is run. The script needs the outputs of the pipeline's first two stages
# (make_data and make_features, see README) and then:
#   1. downloads the release weights and checks them against SHA256SUMS;
#   2. (Tables 2-3 only) chunks and featurises Rini's files with tools/reproduce_tables/rini_features.m;
#   3. scores with tools/reproduce_tables/score.py (x-vector + saved LDA + cosine);
#   4. computes EERs with the bundled BOSARIS toolkit (tools/reproduce_tables/eer.m);
#   5. prints the tables as Markdown and writes them to <out-dir>/tables.md.
#
# Usage: tools/reproduce_tables.sh [options]
#   --pipeline-root DIR  directory holding features/ from make_features (default: this repository)
#   --rini-dir DIR       directory with Rini's <subject>_timit_hear_only_<date>_<time>.mat files;
#                        omit to produce Table 1 only
#   --out-dir DIR        output directory (default: <repo>/reproduce_tables_out)
#   --weights-dir DIR    where the release files are downloaded (default: <out-dir>/release_v0.1.0)
#   --exp NAME           experiment name (default: dataset_1_9_channels_5_A, the default configuration.m)
# Environment variables (each option can also be given this way):
#   PYTHON       Python 3.6 interpreter with requirements.txt installed (required)
#   OCTAVE       Octave command (default: octave-cli)
#   PIPELINE_ROOT, RINI_DIR, OUT_DIR, WEIGHTS_DIR, EXP_NAME
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPERS="$REPO_ROOT/tools/reproduce_tables"
GITHUB_REPO="mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques"
RELEASE_TAG="v0.1.0"
KERAS_FILE="modified_xvector_dataset1_keras.model"
LDA_FILE="modified_xvector_dataset1_lda.bin"

PIPELINE_ROOT="${PIPELINE_ROOT:-$REPO_ROOT}"
RINI_DIR="${RINI_DIR:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reproduce_tables_out}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
EXP_NAME="${EXP_NAME:-dataset_1_9_channels_5_A}"
OCTAVE="${OCTAVE:-octave-cli}"

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --pipeline-root) PIPELINE_ROOT="$2"; shift 2 ;;
    --rini-dir)      RINI_DIR="$2"; shift 2 ;;
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --weights-dir)   WEIGHTS_DIR="$2"; shift 2 ;;
    --exp)           EXP_NAME="$2"; shift 2 ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done
WEIGHTS_DIR="${WEIGHTS_DIR:-$OUT_DIR/release_$RELEASE_TAG}"

# ---- checks ----
[ -n "${PYTHON:-}" ] || die "set PYTHON to the Python 3.6 interpreter from README step 3 (TensorFlow 1.9, Keras 2.2.2, scikit-learn 0.19.2)"
"$PYTHON" -c 'import tensorflow, keras, sklearn, scipy' 2>/dev/null \
  || die "$PYTHON cannot import tensorflow/keras/sklearn/scipy; install requirements.txt into it"
SKL=$("$PYTHON" -c 'import sklearn; print(sklearn.__version__)')
[ "$SKL" = "0.19.2" ] || echo "WARNING: scikit-learn $SKL; the release LDA pickle was written with 0.19.2" >&2
command -v ${OCTAVE%% *} >/dev/null 2>&1 || die "Octave not found ('$OCTAVE'); set OCTAVE to your octave-cli"
PIPELINE_ROOT="$(cd "$PIPELINE_ROOT" && pwd)"
FEATURES="$PIPELINE_ROOT/features/$EXP_NAME/15/1"
if [ ! -d "$FEATURES/train" ] || [ ! -d "$FEATURES/test" ]; then
  cat >&2 <<MSG
ERROR: features not found at $FEATURES/{train,test}.
Run the first two pipeline stages (make_data, make_features; no training) first, e.g. from src/:
  export EEG_DATA_DIR=/path/to/IITM_Task_Independent_EEG_Dataset
  octave-cli --eval "more off; save_default_options('-v7'); addpath('octave_compat'); addpath(genpath('library')); config = configuration; make_data(config); make_features(config);"
or point --pipeline-root at a checkout where they were already run.
MSG
  exit 1
fi
if [ -n "$RINI_DIR" ]; then
  [ -d "$RINI_DIR" ] || die "Rini directory not found: $RINI_DIR"
  RINI_DIR="$(cd "$RINI_DIR" && pwd)"
fi
mkdir -p "$OUT_DIR" "$WEIGHTS_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"; WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
START=$(date +%s)

# ---- 1. release weights ----
echo "[1/5] Release $RELEASE_TAG weights -> $WEIGHTS_DIR"
(
  cd "$WEIGHTS_DIR"
  if [ -f SHA256SUMS ] && sha256sum --quiet -c SHA256SUMS >/dev/null 2>&1; then
    echo "  already downloaded, checksums OK"
  else
    for f in SHA256SUMS "$KERAS_FILE" "$LDA_FILE"; do
      curl -fsSL --retry 3 -o "$f" "https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/$f" \
        || die "download of $f failed (alternatively: gh release download $RELEASE_TAG -R $GITHUB_REPO -D $WEIGHTS_DIR)"
    done
    sha256sum -c SHA256SUMS || die "checksum mismatch in $WEIGHTS_DIR"
  fi
)

# ---- 2. Rini features ----
RINI_FEATS=""
if [ -n "$RINI_DIR" ]; then
  echo "[2/5] Chunking and featurising Rini's files from $RINI_DIR"
  RINI_FEATS="$OUT_DIR/rini_features"
  mkdir -p "$RINI_FEATS"
  (cd "$REPO_ROOT/src" && RT_RINI_DIR="$RINI_DIR" RT_OUT_DIR="$RINI_FEATS" $OCTAVE -q --no-window-system "$HELPERS/rini_features.m" 2> >(grep -v "shadows a core library function" >&2)) \
    || die "rini_features.m failed"
else
  echo "[2/5] No Rini directory given (--rini-dir / RINI_DIR): Tables 2-3 will be skipped."
fi

# ---- 3. scoring ----
echo "[3/5] Scoring with the release model"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}" "$PYTHON" "$HELPERS/score.py" \
  --features "$FEATURES" --keras-weights "$WEIGHTS_DIR/$KERAS_FILE" --lda "$WEIGHTS_DIR/$LDA_FILE" \
  --rini-features "$RINI_FEATS" --out-dir "$OUT_DIR" 2> >(grep -v -e '^Using TensorFlow backend' >&2) \
  || die "score.py failed"

# ---- 4. EER ----
echo "[4/5] EER (BOSARIS rocch + rocch2eer)"
(cd "$REPO_ROOT/src" && RT_SCORES="$OUT_DIR/scores.mat" RT_EER_OUT="$OUT_DIR/eer.txt" $OCTAVE -q --no-window-system "$HELPERS/eer.m" 2> >(grep -v "shadows a core library function" >&2)) \
  || die "eer.m failed"

# ---- 5. tables ----
echo "[5/5] Tables (also written to $OUT_DIR/tables.md)"
echo
"$PYTHON" "$HELPERS/make_tables.py" "$OUT_DIR/acc.json" "$OUT_DIR/eer.txt" "$OUT_DIR/tables.md"
echo "Done in $(( $(date +%s) - START )) s."
