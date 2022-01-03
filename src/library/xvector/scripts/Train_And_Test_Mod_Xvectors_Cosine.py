import sys
temp_dir = sys.argv[1]
save_dir = sys.argv[2]
test_only = int(sys.argv[3])

# Seed value
# Apparently you may use different seed values at each stage
seed_value= int(sys.argv[4])

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import os
import shutil
from x_vector_models import get_modified_x_vector_model

from keras import losses
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy

import scipy.io as sio

import hdf5storage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pickle

# fix random seed for reproducibility
from tqdm import tqdm

dataset = hdf5storage.loadmat(temp_dir+'/subject_id_dataset_for_keras.mat')




def mixed_mse_cross_entropy_loss(y_true, y_pred):
    return 0.8 * losses.categorical_crossentropy(y_true, y_pred) + 0.2 * losses.mean_squared_error(y_true,y_pred)


get_custom_objects().update({"mixed_loss": mixed_mse_cross_entropy_loss})

# split into input (X) and output (Y) variables
trainData = dataset['train_data']
trainLabel = dataset['train_label']
valData = dataset['val_data']
valLabel = dataset['val_label']
testData = dataset['test_data']
testLabel = dataset['test_label']
hiddenLayerConfig = dataset['hidden_layer_size']


num_channels = trainData.shape[1]
train_data = np.moveaxis(trainData, 2, -1)
no_of_examples = trainLabel.shape[1]
randperm = np.random.permutation(no_of_examples)
trainLabel = trainLabel[:, randperm]
train_data = train_data[randperm,:,:,:]
train_label = to_categorical(np.squeeze(trainLabel - 1), num_classes=np.unique(trainLabel).shape[0])





val_data = np.moveaxis(valData, 2, -1)
val_label =  to_categorical(np.squeeze(valLabel - 1), num_classes=train_label.shape[1])


test_data = np.moveaxis(testData, 2, -1)

test_label =  to_categorical(np.squeeze(testLabel - 1), num_classes=train_label.shape[1])





if test_only == 0 :
    model = get_modified_x_vector_model(train_data, train_label, num_channels, hiddenLayerConfig)
    adam_opt = Adam(lr=0.001, clipvalue=1)
    model.compile(loss=losses.categorical_crossentropy, optimizer=adam_opt,
                  metrics=['accuracy', top_k_categorical_accuracy])
    model.summary()
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(save_dir+'/keras.model', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00000000001, verbose=1)

    #Fit the model
    model.fit(train_data, train_label,validation_data=(val_data, val_label), epochs=1, batch_size=16, verbose=2,
          callbacks=[early_stopping, model_checkpoint, reduce_lr])
    model.load_weights(save_dir+'/keras.model')
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('x_vector').output)
else:
    intermediate_layer_model = get_modified_x_vector_model(train_data, train_label, num_channels, hiddenLayerConfig, forTesting=True)
    adam_opt = Adam(lr=0.001, clipvalue=1)
    intermediate_layer_model.compile(loss=losses.categorical_crossentropy, optimizer=adam_opt,
                  metrics=['accuracy', top_k_categorical_accuracy])
    intermediate_layer_model.summary()
    intermediate_layer_model.load_weights(save_dir+'/keras.model',by_name=True)

trainDataProjection = intermediate_layer_model.predict(train_data)
valDataProjection = intermediate_layer_model.predict(val_data)
testDataProjection = intermediate_layer_model.predict(test_data)





trainLabel =  np.squeeze(np.transpose(trainLabel),axis=1)
valLabel =  np.squeeze(np.transpose(valLabel),axis=1)
testLabel =  np.squeeze(np.transpose(testLabel),axis=1)
if test_only == 0:
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit_transform(trainDataProjection, trainLabel)
    pickle.dump(clf, open(save_dir+'/ldaparms.bin', 'wb'))
else:
    clf = pickle.load(open(save_dir+'/ldaparms.bin','rb'))

accumulated_train_data=[]
accumulated_train_label=[]
for i in tqdm(np.unique(trainLabel)):
    subject_data=np.squeeze(train_data[np.where(trainLabel==i),:,:,:],axis=0)
    subject_data_list = []
    for j in range(subject_data.shape[0]):
        subject_data_list.append(subject_data[j,:,:,:])
    subject_data=np.concatenate(subject_data_list, axis=1)
    accumulated_train_data.append(intermediate_layer_model.predict(np.expand_dims(subject_data,axis=0)))
    accumulated_train_label.append(i)

accumulated_train_data = np.concatenate(accumulated_train_data, axis=0)
accumulated_train_label = np.array(accumulated_train_label)


sio.savemat('tmp/sub_space_matrix_befor_lda.mat',{
    'trainSubSpaceVectors': accumulated_train_data,
    'valSubSpaceVectors': valDataProjection,
    'testSubSpaceVectors': testDataProjection,
    'trainLabel': trainLabel,
    'valLabel': valLabel,
    'testLabel': testLabel,
})

accumulated_train_data = clf.transform(accumulated_train_data)
valDataProjection = clf.transform(valDataProjection)
testDataProjection = clf.transform(testDataProjection)

#for ch in range(0,num_channels):
#    test_data_ch_wise[ch] = clf.transform(test_data_ch_wise[ch])



print("Computing Cosine Kernal")

TrainDataKernel = cosine_similarity(accumulated_train_data, accumulated_train_data)
ValDataKernel = cosine_similarity(valDataProjection, accumulated_train_data)
TestDataKernel = cosine_similarity(testDataProjection, accumulated_train_data)
trainLabel = accumulated_train_label

sio.savemat('tmp/sub_space_matrix.mat',{
    'trainSubSpaceVectors': accumulated_train_data,
    'valSubSpaceVectors': valDataProjection,
    'testSubSpaceVectors': testDataProjection,
    'trainLabel': trainLabel,
    'valLabel': valLabel,
    'testLabel': testLabel,
})


print("Cosine Similarity testing")
predictedLabels = np.argmax(TestDataKernel, axis=1)
test_pred = predictedLabels+1
print(accuracy_score(testLabel,predictedLabels) * 100)
testScore = TestDataKernel

predictedLabels = np.argmax(ValDataKernel, axis=1)
val_pred = predictedLabels+1
valScore = ValDataKernel
print(accuracy_score(valLabel,val_pred) * 100)


trainLabel = np.array(trainLabel, dtype=np.int32)
valLabel = np.array(valLabel, dtype=np.int32)
testLabel = np.array(testLabel, dtype=np.int32)

print("Collecting Target and Non-Target Scores")
target_test_scores = []
non_target_test_scores = []
target_val_scores = []
non_target_val_scores = []
indices = np.arange(len(np.unique(trainLabel)))
for i in tqdm(np.unique(trainLabel)):
    scores = np.squeeze(testScore[np.where(testLabel == i), :], axis=0)
    target_scores = scores[:, i-1]
    non_target_scores = np.squeeze(scores[:, indices!=i-1]).flatten()
    target_test_scores.append(target_scores)
    non_target_test_scores.append(non_target_scores)

    scores = np.squeeze(valScore[np.where(valLabel == i), :], axis=0)
    target_scores = scores[:, i-1]
    non_target_scores = np.squeeze(scores[:,  indices!=i-1]).flatten()
    target_val_scores.append(target_scores)
    non_target_val_scores.append(non_target_scores)

target_test_scores = np.concatenate(target_test_scores, axis=0)
non_target_test_scores = np.concatenate(non_target_test_scores, axis=0)
target_val_scores = np.concatenate(target_val_scores, axis=0)
non_target_val_scores = np.concatenate(non_target_val_scores, axis=0)


print("Collecting Normalised Target and Non-Target Scores")

finalTestScore=[]
for i in tqdm(np.unique(trainLabel)):
    score = testScore[:, i-1]
    imposter_score = testScore[:, indices!=i-1]
    mean_is = np.mean(imposter_score,axis=1)
    std_is = np.std(imposter_score,axis=1)
    score = (score - mean_is)/std_is
    finalTestScore.append(score)
finalTestScore = np.transpose(np.array(finalTestScore))


finalValScore=[]
for i in tqdm(np.unique(trainLabel)):
    score = valScore[:, i-1]
    imposter_score = valScore[:, indices!=i-1]
    mean_is = np.mean(imposter_score,axis=1)
    std_is = np.std(imposter_score,axis=1)
    score = (score - mean_is)/std_is
    finalValScore.append(score)
finalValScore = np.transpose(np.array(finalValScore))


target_test_norm_scores=[]
non_target_test_norm_scores=[]
target_val_norm_scores=[]
non_target_val_norm_scores=[]
for i in tqdm(np.unique(trainLabel)):
    scores = np.squeeze(finalTestScore[np.where(testLabel == i), :], axis=0)
    target_norm_scores = scores[:, i-1]
    non_target_norm_scores = np.squeeze(scores[:, indices!=i-1]).flatten()
    target_test_norm_scores.append(target_norm_scores)
    non_target_test_norm_scores.append(non_target_norm_scores)

    scores = np.squeeze(finalValScore[np.where(valLabel == i), :], axis=0)
    target_norm_scores = scores[:, i-1]
    non_target_norm_scores = np.squeeze(scores[:,  indices!=i-1]).flatten()
    target_val_norm_scores.append(target_norm_scores)
    non_target_val_norm_scores.append(non_target_norm_scores)

target_test_norm_scores = np.concatenate(target_test_norm_scores, axis=0)
non_target_test_norm_scores = np.concatenate(non_target_test_norm_scores,axis=0)
target_val_norm_scores = np.concatenate(target_val_norm_scores, axis=0)
non_target_val_norm_scores = np.concatenate(non_target_val_norm_scores,axis=0)


sio.savemat(save_dir+'/predicted_labels.mat',{
    'test_score':finalTestScore,
    'val_score':finalValScore,
    'predicted_labels': test_pred,
    'predicted_val_labels': val_pred,
    'target_test_scores': target_test_scores,
    'non_target_test_scores': non_target_test_scores,
    'target_val_scores': target_val_scores,
    'non_target_val_scores': non_target_val_scores,
    'target_test_norm_scores': target_test_norm_scores,
    'non_target_test_norm_scores': non_target_test_norm_scores,
    'target_val_norm_scores': target_val_norm_scores,
    'non_target_val_norm_scores': non_target_val_norm_scores,
    })
print('done')
