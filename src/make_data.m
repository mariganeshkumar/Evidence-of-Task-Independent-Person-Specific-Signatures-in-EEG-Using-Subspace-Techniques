function [] = make_data(config)
    path=genpath('library');
    addpath(path);
    config = configuration;
    channels = config.lobe_map{config.lobe};
    for c = 1:length(channels)
        mff_channel_names{c} = ['E',num2str(channels(c))];
    end
    disp(mff_channel_names)
    data_dir =  config.data_dir;
    splited_dir = config.splited_dir;
    split = config.split;
    disp(data_dir)
    experiments = get_all_sub_dir(data_dir);
    disp(experiments)
    subject_wise_data=strcat(config.tmp_dir,'/subject_wise_data');
    protcol_wise_data=config.data_dir;
    for i = 1:length(experiments)
        recordings = dir([data_dir,'/',experiments{i},'/**/*.mat']);
        base_folder= {recordings(:).folder};
        recordings = {recordings(:).name};
        for j = 1:length(recordings)
            subject_split = strsplit(recordings{j},'_');
            subject_name = lower(subject_split{1});
            session_identifier=[subject_split{end-1}];
            session_time = strsplit(subject_split{end},".");
            session_time = session_time{1};
            subject_wise_folder=strcat(subject_wise_data,'/',subject_name,'/',session_identifier,'/',experiments{i});
            mkdir(subject_wise_folder);
            newFileName= [subject_name,'_',experiments{i},'_',session_identifier,'_',session_time,'.mat'];
            system(['ln -s ',base_folder{j},'/',recordings{j},' ',subject_wise_folder,'/',newFileName]);
        end
    end
    disp('created subject wise links');
    
    
    subjects = get_all_sub_dir(subject_wise_data);
    for i = 1:length(subjects)
        session = get_all_sub_dir([...
                subject_wise_data,...
                '/',subjects{i}
                ]);
        no_of_sessions=length(session);
        if no_of_sessions == 2
            train_sessions = 1;
        else
            train_sessions = floor(no_of_sessions * 0.7);
        end
        for j = 1:no_of_sessions
            experiments = get_all_sub_dir([...
                subject_wise_data,...
                '/',subjects{i},...
                '/',session{j}
                ]);
            for k = 1:length(experiments)
                recordings_dir = [...
                        subject_wise_data,...
                        '/',subjects{i},...
                        '/',session{j},...
                        '/',experiments{k},...
                        ];
                disp(recordings_dir)
                recordings = dir([recordings_dir,'/*.mat']);
                recordings = {recordings(:).name};
                disp(recordings)            
                for l = 1:length(recordings)
                    split_str = strsplit(recordings{l},'_');
                    session_time = split_str{end};
                    session_time = strsplit(session_time,'.');
                    session_time = session_time{1};
                    %if  j == 1+train_sessions
                    %    divIntoValTestChunks([recordings_dir,'/',recordings{l}],[splited_dir,'/', num2str(split),'/'],config.val_per,subjects{i}, session{j}, experiments{k}, session_time, split, config.lobe_map{config.lobe}, mff_channel_names);
                    if j <= train_sessions
                        divIntoChunks([recordings_dir,'/',recordings{l}],[splited_dir,'/', num2str(split),'/'], 'train',subjects{i}, session{j}, experiments{k}, session_time, split, config.lobe_map{config.lobe}, mff_channel_names);
                    else
                        divIntoValTestChunks([recordings_dir,'/',recordings{l}],[splited_dir,'/', num2str(split),'/'], config.val_per,subjects{i}, session{j}, experiments{k}, session_time, split, config.lobe_map{config.lobe}, mff_channel_names);
                    end
                end
            end
        end
    end
end
    
    
    
    
    
function [all_session_data,test_data] = divIntoChunks(filename, save_dir, data_type ,subject_ind, session_name, experiment_name, sess_time, split, channels, mff_channels)
    
    %Give the complete path of the mat file in "filename" and the number of
    %seconds of which you want the chunk to be in split..
    
    %The function returns train cell and test cell. Each chunk will be randomly
    %added to either train cell or test cell.
    
    data_struct = load(filename);
    try
        data_fields = fieldnames(data_struct);
        data_struct1 = data_struct.(data_fields{1});
        data_fields = fieldnames(data_struct1);
        all_chan_data = data_struct1.(data_fields{1});
        data_struct = data_struct1;
    catch
        data_fields = fieldnames(data_struct);
        all_chan_data = data_struct.(data_fields{1});
    end
    sampling_rate = 250;
    cleanchanlocs = data_struct.cleanchanlocs;
    current_data = all_chan_data(channels,:);
    savedir=strcat(save_dir,'/',data_type,'/',subject_ind,'/',session_name,'/',experiment_name,'/')
    disp(savedir)
    mkdir(savedir);
    current_data = current_data(:,250*5:end-250*5); %% Sampling rate hardcoded
    current_data = current_data - mean(current_data,2);  %DC shift
    std_data = std(current_data,[],2);
    tot_chunks = floor(size(current_data,2)/(split*250)); %%todo: ignore intial and final 5 secs
    for i=0:tot_chunks-1
        data = current_data(:,1+i*250*split:(i+1)*250*split);
        data = data - mean(data,2);
        save([save_dir,'/',data_type,'/',subject_ind,'/',session_name,'/',experiment_name,'/',num2str(i), sess_time, '.tmp.mat'],'data','sampling_rate','cleanchanlocs'); %% tmp files will be ignored in sync and git
    end
end
    
    
    
    function [all_session_data,test_data] = divIntoValTestChunks(filename, save_dir, per ,subject_ind, session_name, experiment_name, sess_time, split, channels, mff_channels)
    
    %Give the complete path of the mat file in "filename" and the number of
    %seconds of which you want the chunk to be in split..
    
    %The function returns train cell and test cell. Each chunk will be randomly
    %added to either train cell or test cell.
    
    data_struct = load(filename);
    try
        data_fields = fieldnames(data_struct);
        data_struct1 = data_struct.(data_fields{1});
        data_fields = fieldnames(data_struct1);
        all_chan_data = data_struct1.(data_fields{1});
        data_struct = data_struct1;
    catch
        data_fields = fieldnames(data_struct);
        all_chan_data = data_struct.(data_fields{1});
    end
    sampling_rate = 250;
    cleanchanlocs = data_struct.cleanchanlocs;
     current_data = all_chan_data(channels,:);
    current_data = current_data(:,250*5:end-250*5); %% Sampling rate hardcoded
    current_data = current_data - mean(current_data,2);  %DC shift
    std_data = std(current_data,[],2);
    tot_chunks = floor(size(current_data,2)/(split*250)); %%todo: ignore intial and final 5 secs
    val_chunks = floor(tot_chunks*per);
    mkdir([save_dir,'/val/',subject_ind,'/',session_name,'/',experiment_name,'/']);
    for i=0:val_chunks-1
        data = current_data(:,1+i*250*split:(i+1)*250*split);
        data = data - mean(data,2);
        save([save_dir,'/val/',subject_ind,'/',session_name,'/',experiment_name,'/',num2str(i), sess_time, '.tmp.mat'],'data','sampling_rate', 'cleanchanlocs'); %% tmp files will be ignored in sync and git
    end
    
    mkdir([save_dir,'/test/',subject_ind,'/',session_name,'/',experiment_name,'/']);
    for i=val_chunks:tot_chunks-1
        data = current_data(:,1+i*250*split:(i+1)*250*split);
        data = data - mean(data,2);
        save([save_dir,'/test/',subject_ind,'/',session_name,'/',experiment_name,'/',num2str(i), sess_time, '.tmp.mat'],'data','sampling_rate','cleanchanlocs'); %% tmp files will be ignored in sync and git
    end  
end