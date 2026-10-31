#!/usr/bin/env bash
# Reproduce the three results tables with the pretrained model from GitHub release v0.1.0.
#
#   Table 1  full Dataset 1 test set: accuracy and EER (vs. the paper's Table IV)
#   Table 2  the 5 held-out recordings that also exist in Rini's files (unprocessed Net
#            Station exports of the same sessions): dataset files vs. Rini's files
#   Table 3  the same, per subject
#
# One command, no data arguments needed:   PYTHON=/path/to/py36/bin/python tools/reproduce_tables.sh
#
# No training is run. Steps:
#   0. data: downloads the dataset zip (~14 GB) and Rini's zip (~562 MB) from Google Drive with
#      gdown, checks their SHA-256, extracts them into the data dir, and runs ONLY the first two
#      pipeline stages (make_data, make_features) with Octave in a working copy of src/.
#      Everything is cached: finished steps are skipped on the next run.
#      --pipeline-root / --rini-dir skip the corresponding download (use existing outputs/files).
#   1. downloads the release weights and checks them against SHA256SUMS;
#   2. chunks and featurises Rini's 5 recordings with tools/reproduce_tables/rini_features.m;
#   3. scores with tools/reproduce_tables/score.py (x-vector + saved LDA + cosine);
#   4. computes EERs with the bundled BOSARIS toolkit (tools/reproduce_tables/eer.m);
#   5. prints the tables as Markdown and writes them to <out-dir>/tables.md.
#
# Usage: tools/reproduce_tables.sh [options]
#   --data-dir DIR       download/extraction/pipeline cache (default: <repo>/reproduce_tables_data)
#   --pipeline-root DIR  use an existing directory holding features/ from make_features
#                        (skips the dataset download, make_data and make_features)
#   --rini-dir DIR       use an existing directory with Rini's <subject>_timit_hear_only_<date>_<time>.mat
#                        files (skips the Rini download)
#   --no-rini            skip Tables 2-3 (Table 1 only)
#   --out-dir DIR        output directory (default: <repo>/reproduce_tables_out)
#   --weights-dir DIR    where the release files are downloaded (default: <out-dir>/release_v0.1.0)
#   --exp NAME           experiment name (default: dataset_1_9_channels_5_A, the default configuration.m)
# Environment variables (each option can also be given this way):
#   PYTHON       Python 3.6 interpreter with requirements.txt installed (required)
#   OCTAVE       Octave command (default: octave-cli)
#   GDOWN        gdown command (default: `gdown` on PATH, else it is pip-installed into a
#                venv <data-dir>/gdown-venv using python3 >= 3.8)
#   DATA_DIR, PIPELINE_ROOT, RINI_DIR, NO_RINI=1, OUT_DIR, WEIGHTS_DIR, EXP_NAME
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPERS="$REPO_ROOT/tools/reproduce_tables"
GITHUB_REPO="mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques"
RELEASE_TAG="v0.1.0"
KERAS_FILE="modified_xvector_dataset1_keras.model"
LDA_FILE="modified_xvector_dataset1_lda.bin"

# ---- Google Drive sources (shared by the repository owner) ----
# Dataset 1: IITM Task-Independent EEG dataset; extracts to IITM_Task_Independent_EEG_Dataset/ (12 task folders)
DATASET_DRIVE_ID="11zm2V5jGNZXs0uOok1Gfcd65SszVoqWF"
DATASET_DRIVE_URL="https://drive.google.com/file/d/${DATASET_DRIVE_ID}/view"
DATASET_ZIP="IITM_Task_Independent_EEG_Dataset.zip"
DATASET_ZIP_SHA256="06fbb788457e5611eef113cba949ae97464bd626ff47f603d51be4aca7bb578e"
# Rini's files (unprocessed Net Station exports of the same sessions);
# extracts to subject_id_7subjects/IITM_Task_Independent_EEG_Dataset/timit_hear_only/
RINI_DRIVE_ID="1XeluL1uVPyars5b8JxtIPYWNVEBYjsYN"
RINI_DRIVE_URL="https://drive.google.com/file/d/${RINI_DRIVE_ID}/view"
RINI_ZIP="subject_id_7subjects.zip"
RINI_ZIP_SHA256="a2a38007493b47162668dfadea252d71e83a21ede6b0f04750fece870346ca51"
RINI_SUBDIR="subject_id_7subjects/IITM_Task_Independent_EEG_Dataset/timit_hear_only"
# The 5 recordings used for Tables 2-3 (must match RECORDINGS in tools/reproduce_tables/score.py)
RINI_FILES=(012100073_timit_hear_only_20180819_040327.mat 014100078_timit_hear_only_20180822_043400.mat
            015100076_timit_hear_only_20180822_080342.mat 016100071_timit_hear_only_20180819_025325.mat
            017000070_timit_hear_only_20180822_010640.mat)

DATA_DIR="${DATA_DIR:-$REPO_ROOT/reproduce_tables_data}"
PIPELINE_ROOT="${PIPELINE_ROOT:-}"
RINI_DIR="${RINI_DIR:-}"
NO_RINI="${NO_RINI:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reproduce_tables_out}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
EXP_NAME="${EXP_NAME:-dataset_1_9_channels_5_A}"
OCTAVE="${OCTAVE:-octave-cli}"

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir)      DATA_DIR="$2"; shift 2 ;;
    --pipeline-root) PIPELINE_ROOT="$2"; shift 2 ;;
    --no-rini)       NO_RINI=1; shift ;;
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
[ -z "$RINI_DIR" ] || [ -z "$NO_RINI" ] || die "--rini-dir and --no-rini are mutually exclusive"
if [ -n "$PIPELINE_ROOT" ]; then
  [ -d "$PIPELINE_ROOT" ] || die "pipeline root not found: $PIPELINE_ROOT"
  PIPELINE_ROOT="$(cd "$PIPELINE_ROOT" && pwd)"
  FEATURES="$PIPELINE_ROOT/features/$EXP_NAME/15/1"
  if [ ! -d "$FEATURES/train" ] || [ ! -d "$FEATURES/test" ]; then
    cat >&2 <<MSG
ERROR: features not found at $FEATURES/{train,test}.
Either drop --pipeline-root (the script then downloads the dataset and runs make_data and
make_features itself), or run those two stages (no training) in that checkout, from src/:
  export EEG_DATA_DIR=/path/to/IITM_Task_Independent_EEG_Dataset/
  octave-cli --eval "more off; save_default_options('-v7'); addpath('octave_compat'); addpath(genpath('library')); config = configuration; make_data(config); make_features(config);"
MSG
    exit 1
  fi
fi
if [ -n "$RINI_DIR" ]; then
  [ -d "$RINI_DIR" ] || die "Rini directory not found: $RINI_DIR"
  RINI_DIR="$(cd "$RINI_DIR" && pwd)"
fi
mkdir -p "$OUT_DIR" "$WEIGHTS_DIR" "$DATA_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"; WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"; DATA_DIR="$(cd "$DATA_DIR" && pwd)"
START=$(date +%s)

# ---- helpers for step 0 ----
free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }
need_space() {  # need_space GB what
  local f; f=$(free_gb "$DATA_DIR")
  [ "$f" -ge "$1" ] || die "$2 needs about $1 GB free in $DATA_DIR, only $f GB available (use --data-dir elsewhere)"
}
GDOWN_CMD=()
setup_gdown() {
  [ ${#GDOWN_CMD[@]} -eq 0 ] || return 0
  if [ -n "${GDOWN:-}" ]; then GDOWN_CMD=($GDOWN); return 0; fi
  if command -v gdown >/dev/null 2>&1; then GDOWN_CMD=(gdown); return 0; fi
  local venv="$DATA_DIR/gdown-venv"
  if [ ! -x "$venv/bin/gdown" ]; then
    echo "  gdown not found; installing it with pip into $venv"
    command -v python3 >/dev/null 2>&1 || die "python3 (>= 3.8) is needed to install gdown; or install gdown yourself and set GDOWN"
    python3 -m venv "$venv" && "$venv/bin/pip" install -q gdown \
      || die "could not pip-install gdown into $venv; install gdown yourself (pip install gdown) and set GDOWN"
  fi
  GDOWN_CMD=("$venv/bin/gdown")
}
sha_ok() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }
fetch_zip() {  # fetch_zip drive_id file sha256 size_gb
  local id="$1" f="$DATA_DIR/$2" sha="$3" gb="$4"
  if sha_ok "$f" "$sha"; then echo "  $2: already downloaded, SHA-256 OK"; return 0; fi
  need_space $(( gb + 1 )) "Downloading $2"
  setup_gdown
  echo "  downloading $2 (~$gb GB) from Google Drive with gdown"
  rm -f "$f.part"
  if ! "${GDOWN_CMD[@]}" "$id" -O "$f.part"; then
    echo "  gdown failed; trying curl (drive.usercontent.google.com, confirm=t)" >&2
    curl -fL --retry 3 -o "$f.part" "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" \
      || die "download of $2 failed (Drive quota or confirm page?). Download it manually from https://drive.google.com/file/d/$id/view into $DATA_DIR/$2 and re-run"
  fi
  sha_ok "$f.part" "$sha" || die "SHA-256 mismatch for $2 (expected $sha; got $(sha256sum "$f.part" | cut -d' ' -f1)); the partial file is $f.part"
  mv "$f.part" "$f"
  echo "  $2: SHA-256 OK"
}
extract_zip() {  # extract_zip file stamp_name size_gb
  local f="$DATA_DIR/$1" stamp="$DATA_DIR/.$2.extracted"
  if [ -f "$stamp" ]; then echo "  $1: already extracted"; return 0; fi
  need_space $(( $3 + 1 )) "Extracting $1"
  echo "  extracting $1"
  unzip -q -o "$f" -d "$DATA_DIR" || die "unzip of $1 failed"
  touch "$stamp"
}

# ---- 0. data ----
echo "[0/5] Data (cache: $DATA_DIR)"
if [ -z "$PIPELINE_ROOT" ]; then
  EEG_DIR="$DATA_DIR/IITM_Task_Independent_EEG_Dataset"
  if [ ! -f "$DATA_DIR/pipeline/.features.done" ]; then
    [ -f "$DATA_DIR/.dataset.extracted" ] || fetch_zip "$DATASET_DRIVE_ID" "$DATASET_ZIP" "$DATASET_ZIP_SHA256" 14
    extract_zip "$DATASET_ZIP" dataset 15
    ntask=$(find "$EEG_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    [ "$ntask" -eq 12 ] || die "expected 12 task folders in $EEG_DIR after extraction, found $ntask (delete $DATA_DIR/.dataset.extracted and $EEG_DIR to re-extract)"
  fi
  # make_data + make_features only (no training), in a working copy of src/ so that the outputs
  # (../splited_data, ../features, ../tmp relative to src/) land in $DATA_DIR/pipeline
  PIPELINE_ROOT="$DATA_DIR/pipeline"
  FEATURES="$PIPELINE_ROOT/features/$EXP_NAME/15/1"
  if [ -f "$PIPELINE_ROOT/.features.done" ]; then
    echo "  make_data / make_features: already done ($PIPELINE_ROOT)"
  else
    for d in splited_data features; do
      [ ! -e "$PIPELINE_ROOT/$d" ] || die "$PIPELINE_ROOT/$d exists from an unfinished run; delete $PIPELINE_ROOT/splited_data and $PIPELINE_ROOT/features and re-run"
    done
    need_space 15 "make_data + make_features"
    mkdir -p "$PIPELINE_ROOT"
    rm -rf "$PIPELINE_ROOT/src"; cp -r "$REPO_ROOT/src" "$PIPELINE_ROOT/src"
    echo "  running make_data and make_features with Octave (no training); this takes a while"
    (cd "$PIPELINE_ROOT/src" && EEG_DATA_DIR="$EEG_DIR/" $OCTAVE -q --no-window-system --eval \
      "more off; page_screen_output(false); save_default_options('-v7'); addpath('octave_compat'); addpath(genpath('library')); config = configuration; make_data(config); make_features(config);" \
      > "$PIPELINE_ROOT/make_data_features.log" 2>&1) \
      || die "make_data / make_features failed; see $PIPELINE_ROOT/make_data_features.log"
    [ -d "$FEATURES/train" ] && [ -d "$FEATURES/test" ] || die "make_features did not produce $FEATURES; see $PIPELINE_ROOT/make_data_features.log"
    touch "$PIPELINE_ROOT/.features.done"
  fi
else
  echo "  using existing pipeline outputs in $PIPELINE_ROOT"
fi
if [ -z "$NO_RINI" ] && [ -z "$RINI_DIR" ]; then
  [ -f "$DATA_DIR/.rini.extracted" ] || fetch_zip "$RINI_DRIVE_ID" "$RINI_ZIP" "$RINI_ZIP_SHA256" 1
  extract_zip "$RINI_ZIP" rini 1
  RINI_DIR="$DATA_DIR/$RINI_SUBDIR"
fi
if [ -n "$RINI_DIR" ]; then
  for f in "${RINI_FILES[@]}"; do [ -f "$RINI_DIR/$f" ] || die "missing Rini file: $RINI_DIR/$f"; done
  echo "  Rini's files: $RINI_DIR (all 5 recordings present)"
fi


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
  echo "[2/5] --no-rini given: Tables 2-3 will be skipped."
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
