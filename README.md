# Evidence of Task-Independent Person-Specific Signatures in EEG Using Subspace Techniques

Code for the paper:

> M. G. Kumar, S. Narayanan, M. Sur and H. A. Murthy, "Evidence of Task-Independent Person-Specific Signatures in EEG Using Subspace Techniques," *IEEE Transactions on Information Forensics and Security*, vol. 16, pp. 2856–2871, 2021.
> [IEEE Xplore](https://ieeexplore.ieee.org/document/9383011) · [arXiv:2007.13517](https://arxiv.org/abs/2007.13517)

This repository contains the **modified x-vector** system from the paper: data splitting, spectral-power feature extraction, x-vector (TDNN) training, and LDA + cosine scoring with accuracy and EER reporting. The pipeline runs on **GNU Octave** (free, recommended) or MATLAB, with the neural network trained in Python (TensorFlow/Keras).

## Contents
- [Results](#results)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the pipeline (GNU Octave, recommended)](#run-the-pipeline-gnu-octave-recommended)
- [Outputs and logs](#outputs-and-logs)
- [Reading the results](#reading-the-results)
- [Runtime and resources](#runtime-and-resources)
- [Configuration](#configuration)
- [Alternative: MATLAB](#alternative-matlab)
- [Evaluating a saved checkpoint](#evaluating-a-saved-checkpoint)
- [Known issues](#known-issues)
- [Citation](#citation)

## Results

Dataset 1 (IITM Task-Independent EEG dataset, 30 subjects), 9 channels, 15 s segments. Accuracy is test-set identification accuracy per segment; EER is the test-set equal error rate.

| System | Source | Accuracy (%) | EER (%) |
|---|---|---|---|
| **Modified x-vector** | **This repository, GNU Octave run (see below)** | **76.71** | **9.50** |
| Modified x-vector | Paper, Table IV | 76.8 | 8.16 |
| Baseline x-vector | Paper, Table IV (code not in this repository) | 67.1 | 11.3 |
| Modified i-vector | Paper, Table IV (code not in this repository) | 85.1 | 5.81 |
| ix-vector | Paper, Table IV (code not in this repository) | 86.4 | 5.02 |

Details of the reproduction run:
- Default `src/configuration.m` (seed 0, one trial, CPU only).
- Training stopped early at epoch 33; the best checkpoint (lowest validation loss) was from epoch 23.
- Split: 4,681 train / 216 validation / 2,598 test segments. The paper used 4,681 / 559 / 2,255 because it holds out 20% for validation, whereas `configuration.m` uses `val_per = 0.10`. This difference is the most likely cause of the gap in EER (9.50% vs 8.16%); accuracy matches the paper.
- Total time about 72 minutes on an 8-core CPU (see [Runtime and resources](#runtime-and-resources)).

TensorFlow on CPU is not bit-for-bit deterministic, so expect small differences from run to run.

## Requirements

- Linux x86_64 with `bash` (the pipeline calls `bash`, `ln -s` and `rm`). Tested on Debian 13.
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), conda or mamba, to create the two environments below.
- About 16 GB RAM and 30 GB free disk (dataset ≈ 14 GB, split data ≈ 13 GB, features ≈ 0.6 GB, models ≈ 55 MB).
- No GPU is needed.

Tested software versions:

| Component | Version |
|---|---|
| GNU Octave (conda-forge) | 10.3.0 |
| Octave `statistics` package (optional) | 1.7.3 |
| Python | 3.6.15 |
| TensorFlow | 1.9.0 |
| Keras | 2.2.2 |
| numpy / scipy / scikit-learn | 1.15.0 / 1.1.0 / 0.19.2 |
| h5py / hdf5storage | 2.8.0 / 0.1.19 |
| matplotlib / tqdm | 3.3.4 / 4.64.1 |

## Setup

### 1. Get the code
```bash
git clone https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques.git eeg-subspace
cd eeg-subspace
```

### 2. GNU Octave environment
```bash
micromamba create -y -n eeg-octave -c conda-forge octave=10.3
```
Optional: install the Octave `statistics` package. If it is not installed (or fails to build), `run_octave.m` uses the small replacements in `src/octave_compat_stats/` instead, so the results are the same.
```bash
micromamba run -n eeg-octave octave-cli --eval "pkg install -forge statistics"
```
The Octave `signal` package is **not** needed; `src/octave_compat/spectrogram.m` provides the one `spectrogram` call the pipeline uses.

### 3. Python 3.6 environment (TensorFlow 1.9 / Keras 2.2.2)
```bash
micromamba create -y -n eeg-py36 -c conda-forge python=3.6 pip
micromamba run -n eeg-py36 pip install -r requirements.txt
```
Note the full path of this environment's Python; the pipeline needs it:
```bash
micromamba run -n eeg-py36 python -c "import sys; print(sys.executable)"
```

### 4. Dataset
The experiments use the **IITM Task-Independent EEG dataset** (30 subjects, 128-channel EEG, 250 Hz, 12 tasks). It is not included in this repository. Request it from the authors (dataset page: https://www.iitm.ac.in/donlab/cbr/eeg_person_id_dataset/).

Place it anywhere; the expected layout is one folder per task, then subject, then session:
```
IITM_Task_Independent_EEG_Dataset/
├── Familiar-and-Unfamiliar-words/
├── Imagining-Binary-Answers/
├── ...                                  (12 task folders in total)
└── Steady-State-Visually-Evoked-Potential/
    └── <subject>/<session_date>/<subject>_<task>_<session_date>_<time>.mat
```
Each `.mat` file contains `current_data` (channels × samples) and `cleanchanlocs`.

Point the pipeline at the dataset with the `EEG_DATA_DIR` environment variable (recommended), or edit `data_dir_128` in `src/configuration.m`. The default is `../IITM_Task_Independent_EEG_Dataset/` relative to `src/`, i.e. a folder of that name in the repository root.

## Run the pipeline (GNU Octave, recommended)

Run from `src/`, with the Octave environment active and `PYTHON` set to the Python 3.6 interpreter from step 3:
```bash
cd src
micromamba activate eeg-octave            # or: export PATH=/path/to/envs/eeg-octave/bin:$PATH
export EEG_DATA_DIR=/path/to/IITM_Task_Independent_EEG_Dataset
export PYTHON=/path/to/envs/eeg-py36/bin/python
export TF_CPP_MIN_LOG_LEVEL=1             # optional: less TensorFlow logging
mkdir -p ../logs
time octave-cli run_octave.m 2>&1 | tee ../logs/run_$(date +%Y%m%d_%H%M).log
```

`run_octave.m` sets up Octave compatibility (MATLAB-readable `.mat` files, headless figures, the shims above) and then runs `run.m`, which:
1. **make_data** splits each subject's sessions into train / validation / test and cuts 15 s segments (skipped if `../splited_data/15` already exists);
2. **make_features** extracts PSD spectrogram (spectral power, 3–30 Hz) features for the 9-channel montage (skipped if `../features/dataset_1_9_channels_5_A/15` already exists);
3. trains the modified x-vector network in Python, extracts x-vectors, applies LDA and cosine scoring, and computes accuracy and EER.

To start from scratch, delete `splited_data/`, `features/`, `model_history/` and `tmp/` in the repository root.

Warnings such as `saving figure failed: print: rendering with fltk toolkit requires visible figure` are expected on a headless machine: the score-distribution plots are skipped, and the EER is still computed.

## Outputs and logs

Paths are relative to the repository root (`<exp>` = `dataset_1_9_channels_5_A`):

| Path | Contents |
|---|---|
| `logs/run_*.log` | Console log, if you used the `tee` command above |
| `splited_data/15/{train,val,test}/` | Segmented data |
| `features/<exp>/15/1/{train,val,test}/` | Feature files |
| `model_history/<exp>/report_test_segment15_Fe_spectral_power_Clas_modified-x-vector_Cosine.txt` | **Test-set report** (final accuracy and EER) |
| `model_history/<exp>/report_val_segment15_Fe_spectral_power_Clas_modified-x-vector_Cosine.txt` | Validation-set report |
| `model_history/<exp>/spectral_power/modified-x-vector_Cosine/1/dnn_models/keras.model` | Best checkpoint (lowest validation loss) |
| `.../dnn_models/keras_last.model` | Checkpoint from the last completed epoch |
| `.../1/15_score.mat` | Predicted labels and target / non-target scores |
| `tmp/<exp>/subject_id_dataset_for_keras.mat` | Data passed to Python (used by `tools/eval_checkpoint.py`) |

## Reading the results

In the console log, read these two lines:
```
Trail Accuracy 76.7129
EER is 9.50
```
They are the test-set accuracy and EER. The same values appear at the end of the test report as `accuracies = 76.7129` and `EERs = 9.5034`.

Do **not** use the first number printed after `Cosine Similarity testing` (for example `1.73...`). The Python script compares 0-based predictions with 1-based labels there, so that number is wrong. The second number after it is the validation accuracy (80.56 in the run above) and is correct. The Octave lines above use the correct labels.

During training, Keras prints one line per epoch with `loss`, `acc`, `val_loss` and `val_acc`, followed by `Epoch 000NN: early stopping` when training ends.

## Runtime and resources

Measured on an 8-core x86_64 CPU with 16 GB RAM, no GPU:
- x-vector training: about 2 minutes per epoch; the run stopped early after 33 of at most 100 epochs.
- Training, testing and scoring together: 72 minutes wall-clock.
- The first run also builds the data split and features (steps 1–2). This was not timed separately; later runs skip these steps.

## Configuration

All settings are in `src/configuration.m`. The values used for the results above are the defaults:

| Setting | Value |
|---|---|
| `classifier` | 2 (`modified-x-vector_Cosine`) |
| `feature` | 1 (`spectral_power`, PSD spectrogram) |
| `lobe` | 6 (9-channel montage: 11, 70, 83, 36, 104, 122, 33, 58, 96) |
| `split` | 15 (15 s segments) |
| `val_per` | 0.10 (the paper used 20%; see [Results](#results)) |
| `win_size`, `overlap`, `nfft`, `lfreq`, `hfreq` | 360 ms, 0, 256, 3 Hz, 30 Hz |
| `hiddenlayers` | `[1024 512 160]` |
| `epochs` | 100 (maximum; early stopping with patience 10) |
| `seed`, `trials`, `GPU_Number` | 0, 1, -1 (CPU) |

The network is trained on raw spectral-power features with Adam (learning rate 1e-3, `clipvalue = 1`). Statistics pooling uses `sqrt(var + 1e-6)` for the standard deviation, which keeps gradients finite.

The modified i-vector and ix-vector entries in `configuration.m` are not implemented in this repository.

## Alternative: MATLAB

MATLAB works as well; the MATLAB path was not re-run for the results above.
- Requirements: MATLAB (R2020b used in the [Dockerfile](Dockerfile)) with the Signal Processing Toolbox and the Statistics and Machine Learning Toolbox, plus the same Python 3.6 environment (steps 3–4 of [Setup](#setup)).
- Run `run.m` from `src/` (not `run_octave.m`, which only works in Octave):
  ```bash
  cd src
  export EEG_DATA_DIR=/path/to/IITM_Task_Independent_EEG_Dataset
  export PYTHON=/path/to/envs/eeg-py36/bin/python
  matlab -batch "run" 2>&1 | tee ../logs/run_matlab.log
  ```
- Outputs and results are read the same way as for Octave.

The [Dockerfile](Dockerfile) builds an image with MATLAB R2020b, the toolboxes and Python 3.6 with `requirements.txt`; a MATLAB license is required to use it.

## Evaluating a saved checkpoint

`tools/eval_checkpoint.py` re-evaluates a trained model without retraining (softmax accuracy on train/val/test, and x-vector + LDA + cosine accuracy on val/test), and writes the scores to `scores.mat`:
```bash
$PYTHON tools/eval_checkpoint.py \
  model_history/dataset_1_9_channels_5_A/spectral_power/modified-x-vector_Cosine/1/dnn_models/keras.model \
  tmp/dataset_1_9_channels_5_A/subject_id_dataset_for_keras.mat \
  eval_out
```
See the docstring at the top of the script for computing the EER from `scores.mat`.

## Known issues

- The first accuracy printed by the Python script is wrong by an off-by-one in the labels (see [Reading the results](#reading-the-results)).
- The console line `Normalised EER is ...` prints the plain test EER, and the validation report's "Normalised EERs" value is computed from test scores.
- `val_per = 0.10` differs from the paper's 20% validation split.

## Citation

```bibtex
@article{kumar2021evidence,
  author  = {Kumar, Mari Ganesh and Narayanan, Shrikanth and Sur, Mriganka and Murthy, Hema A.},
  title   = {Evidence of Task-Independent Person-Specific Signatures in {EEG} Using Subspace Techniques},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {16},
  pages   = {2856--2871},
  year    = {2021},
  doi     = {10.1109/TIFS.2021.3067998}
}
```

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
