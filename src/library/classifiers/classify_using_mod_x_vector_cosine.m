function [result] = classify_using_mod_x_vector_cosine(config,allData)


models_dir =  [allData.modelSaveDir,'/',config.dnn_model_files];
system(['rm -rf ', config.tmp_dir]);
mkdir(config.tmp_dir);
train_data = allData.train_data;
test_data = allData.test_data;
val_data = allData.val_data;

disp('collecting train data');
train_data_cell={};
train_label={};
parfor i=1:length(train_data)
    ind=1;
    train_data_s=[];
    train_label_s=[];
    for j=1:length(train_data{i})
        for k=1:length(train_data{i}{j})
            recording_data=train_data{i}{j}{k};
            train_data_s(ind,:,:,:) = recording_data;
            train_label_s(ind)=i;
            ind = ind+1;
        end
    end
    train_data_cell{i}=train_data_s;
    train_label{i}=train_label_s;
end
train_data_mat=cat(1,train_data_cell{:});
train_label = cat(2,train_label{:});

disp('preparing val data')
val_data_cell = {};
val_label={};
test_data_cell = {};
test_label = {};

parfor i=1:length(test_data)
    test_ind=1;
    test_data_s=[];
    test_label_s=[];
    for j=1:length(test_data{i})
        for k=1:length(test_data{i}{j})
            recording_data=test_data{i}{j}{k};
            test_data_s(test_ind,:,:,:) = recording_data;
            test_label_s(test_ind)=i;
            test_ind = test_ind+1;
        end
    end
    test_data_cell{i} = test_data_s;
    test_label{i} = test_label_s;
end


parfor i=1:length(val_data)
    val_ind=1;
    val_data_s=[];
    val_label_s=[];
    for j=1:length(val_data{i})
        for k=1:length(val_data{i}{j})
            recording_data=val_data{i}{j}{k};
            val_data_s(val_ind,:,:,:) = recording_data;
            val_label_s(val_ind)=i;
            val_ind = val_ind+1;
        end
    end
    val_data_cell{i} = val_data_s;
    val_label{i} = val_label_s;
end


val_data_mat = cat(1, val_data_cell{:});
val_label = cat(2, val_label{:});
test_data_mat = cat(1, test_data_cell{:});
test_label = cat(2, test_label{:});

try
GPU_no = config.GPU_Number;
catch
GPU_no = 0;
end

run_ann(config.seed, config.hiddenlayers, config.tmp_dir, train_data_mat, train_label,val_data_mat, val_label, test_data_mat, test_label, models_dir, config.test_only, GPU_no);
if (config.make_sub_space_plot==1)
    channels = config.lobe_map{config.lobe};
    make_plots_channel_contrib([models_dir,'/cosine_sim.mat'],[allData.plotSaveDir,'/',num2str(config.split)], allData.subjects, channels)
 
    make_plots_from_sub_space('tmp/',[allData.plotSaveDir,'/',num2str(config.split)], allData.sessionInfo)
end
data = load([models_dir,'/predicted_labels.mat']);

test_score = double(data.test_score);
val_score = double(data.val_score);


result.test_predicted_label = double(data.predicted_labels);
result.val_predicted_label= double(data.predicted_val_labels);
result.val_gt_label = val_label;
result.test_gt_label = test_label;
subject_wise_target_scores={};
subject_wise_non_target_scores={};
parfor i = 1: length(test_data)
    scores = test_score(result.test_gt_label==i,:);
    subject_wise_target_scores{i} = reshape(scores(:,i),1,[]);
    scores(:,i)=[];
    subject_wise_non_target_scores{i} =  reshape(scores,1,[]);
    %scores = test_score(result.test_gt_label~=i,i);
    %subject_wise_non_target_scores{i} =  [subject_wise_non_target_scores{i}, reshape(scores,1,[])];
end
result.target_test_scores = double(data.target_test_scores);
result.non_target_test_scores = double(data.non_target_test_scores);
result.target_val_scores = double(data.target_val_scores);
result.non_target_val_scores = double(data.non_target_val_scores);
result.target_test_norm_scores = double(data.target_test_norm_scores);
result.non_target_test_norm_scores = double(data.non_target_test_norm_scores);
result.target_val_norm_scores = double(data.target_val_norm_scores);
result.non_target_val_norm_scores = double(data.non_target_val_norm_scores);
result.subject_wise_target_scores = subject_wise_target_scores;
result.subject_wise_non_target_scores = subject_wise_non_target_scores;
end

function [] = run_ann(seed, hidden_layer_size, tmp_dir, train_data, train_label ,val_data, val_label, test_data, test_label, save_dir, only_test, GPU_Number)
disp('writing files for X-vector training')
save([tmp_dir,'/subject_id_dataset_for_keras.mat'],'train_data','train_label','val_data', 'val_label','test_data','test_label','hidden_layer_size','-v7.3')
clear 'train_data' 'train_label' 'test_data' 'test_data' 'val_label' 'val_data' 'test_label';
disp(['bash library/xvector/scripts/Train_And_Test_Mod_Xvectors_Cosine.bash ',tmp_dir, ' ', save_dir, ' ', num2str(only_test), ' ', num2str(seed), ' ', num2str(GPU_Number)])
system(['bash library/xvector/scripts/Train_And_Test_Mod_Xvectors_Cosine.bash ',tmp_dir, ' ', save_dir, ' ', num2str(only_test) , ' ', num2str(seed),' ', num2str(GPU_Number)])
end