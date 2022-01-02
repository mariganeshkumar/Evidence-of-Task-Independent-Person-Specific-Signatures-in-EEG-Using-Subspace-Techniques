function [val_eer,test_eer] = train_and_test_eeg_biometrics(config)
path=genpath('library');
addpath(path);

if ~exist('config','var')
    config = configuration;
end
disp(config)
feature_dir = config.features_dir;

feature_code = [num2str(config.split),...
    '/',num2str(config.feature)];

val_predicted_label =  cell([1 config.trials]);
val_gt_label = cell([1 config.trials]);

val_accuracy = zeros([1 config.trials]);
val_eer = zeros([1 config.trials]);
val_norm_eer = zeros([1 config.trials]);

test_predicted_label = cell([1 config.trials]);
test_gt_label = cell([1 config.trials]);

test_accuracy = zeros([1 config.trials]);
test_eer = zeros([1 config.trials]);
test_norm_eer = zeros([1 config.trials]);


freq_dim = 0;

%get all the subjects
subjects = get_all_sub_dir([feature_dir,'/',feature_code,'/train/']);

datasets={'train','val','test'};
for t=1:config.trials
    
    for i = 1:length(subjects)
        disp(['Loading data for subject: ',num2str(i),'/',subjects{i}]);
        for h = 1:length(datasets)
            subject_dir=[feature_dir,'/',feature_code,'/',datasets{h},'/',subjects{i}]; 
            sessions = get_all_sub_dir([subject_dir]);
            for j = 1:length(sessions)
                session_dir = [subject_dir,'/',sessions{j}];
                experiments = get_all_sub_dir(session_dir);
                trail_ind=1;
                for k = 1:length(experiments)
                    recordings = dir([...
                        session_dir,...
                        '/',experiments{k},...
                        '/*.mat'
                        ]);
                    recordings = {recordings(:).name};
                    for l = 1:length(recordings)
                        data = load([...
                            session_dir,...
                            '/',experiments{k},...
                            '/',recordings{l}
                            ]);
                        data=data.feature;
                        freq_dim = size(data,2);
                        if strcmp(datasets{h},'train') == 1
                            allData.ubm_data{i}{j}{trail_ind} =  data;
                            allData.train_data{i}{j}{trail_ind} = data;
                            allData.sessionInfo{i}=length(sessions);
                        elseif strcmp(datasets{h},'val')
                            allData.val_data{i}{j}{trail_ind} =  data;
                        else
                            allData.test_data{i}{j}{trail_ind} =  data;
                        end
                        trail_ind=trail_ind+1;
                    end
                end
            end
        end
    end
    
    allData.freq_dim = freq_dim;
    allData.subjects = subjects;
    allData.num_channels = length(config.lobe_map{config.lobe});
    modelSaveDir = [config.history_save,...
        '/',config.feature_name{config.feature},...
        '/',config.classifier_name{config.classifier},...
        '/', num2str(t)];
    allData.modelSaveDir = modelSaveDir;
    allData.plotSaveDir = modelSaveDir;
    %disp(['Total Number Of Intersession Subjects:', num2str(intersession_subjects)]);
    %disp(['Total Number Of Sessions:', num2str(sum(sessionInfo))]);
    [results]=config.classifier_function{config.classifier}(config,allData);
    save([modelSaveDir,'/',num2str(config.split),'_score.mat'],'results');
    val_predicted_label{t}=results.val_predicted_label;
    val_gt_label{t}=results.val_gt_label;
    test_predicted_label{t} = results.test_predicted_label;
    test_gt_label{t} = results.test_gt_label;
    
    clear allData;
    cm = confusionmat(test_predicted_label{t}, test_gt_label{t});
    test_accuracy(t) = sum(diag(cm))/sum(sum(cm)) * 100;
    
    disp(['Trail Accuracy ', num2str(test_accuracy(t))]);
    cm = confusionmat(val_predicted_label{t}, val_gt_label{t});
    val_accuracy(t) = sum(diag(cm))/sum(sum(cm)) * 100;
    save_dir=[modelSaveDir,'/',num2str(config.split),'_'];
    
    
    test_eer(t) = plot_score_distribution([save_dir,'test'],results.target_test_scores, results.non_target_test_scores);
    val_eer(t) = plot_score_distribution([save_dir,'val'],results.target_val_scores, results.non_target_val_scores);
    fprintf('EER is %.2f\n', test_eer(t));
    
    
    test_norm_eer(t) = plot_score_distribution([save_dir,'test_norm'],results.target_test_norm_scores, results.non_target_test_norm_scores);
    val_norm_eer(t) = plot_score_distribution([[save_dir,'val_norm']],results.target_test_norm_scores, results.non_target_test_norm_scores);
    fprintf('Normalised EER is %.2f\n', test_eer(t));
    
end
val_predicted_label;
val_predicted_label = cell2mat(val_predicted_label);
val_gt_label = cell2mat(val_gt_label);

test_predicted_label = cell2mat(test_predicted_label);
test_gt_label = cell2mat(test_gt_label);

result_str = strcat(config.history_save,...
    '/report_val_segment',num2str(config.split),'_Fe_',...
    config.feature_name{config.feature},'_Clas_',...
    config.classifier_name{config.classifier},'.txt');


WriteReport(result_str, val_accuracy, val_predicted_label,...
    val_gt_label,val_eer,val_norm_eer,subjects);

result_str = strcat(config.history_save,...
    '/report_test_segment',num2str(config.split),'_Fe_',...
    config.feature_name{config.feature},'_Clas_',...
    config.classifier_name{config.classifier},'.txt');


WriteReport(result_str, test_accuracy, test_predicted_label,...
    test_gt_label,test_eer,test_norm_eer,subjects);

%system(['rm -rf ',config.tmp_dir]);
end