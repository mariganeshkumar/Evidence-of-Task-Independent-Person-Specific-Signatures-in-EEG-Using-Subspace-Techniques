# Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques
Implementation of architecures proposed in https://ieeexplore.ieee.org/document/9383011

## Major Requirements
- Matlab
   - Signal Procresing Toolbox
- Python (v3.6)
  -  Keras (v2.2.2)
  -  tensorflow (v1.9.0)

## Instruction to train the models dicussed in paper
1. Install all major requirement and python packages given in [requirements.txt](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/requirements.txt)
2. Clone the respository
3. Request the for data and download the same from [here](https://www.iitm.ac.in/donlab/cbr/eeg_person_id_dataset/)
4. Edit the [configuration](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/src/configuration.m) file to reflect the location of downloaded data
   - The configuration file can also be used to change classifiers (modified-i-vector (**WIP**), modified-x-vector and ix-vector (**WIP**))
5. Run the [Run.m](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/src/run.m) matlab script. This script with do the following
   - Split the diven data into train, validation and text
   - Extract features as per the [configuration](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/src/configuration.m)
   - Train the classifier as per the [configuration](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/src/configuration.m)
   - Report the obtained results on validation and test data

**Note:** The repository also includes a [docker file](https://github.com/mariganeshkumar/Evidence-of-Task-Independent-Person-Specific-Signatures-in-EEG-Using-Subspace-Techniques/blob/main/Dockerfile) which can be used to build a docker image with all requirements pre-installed.
