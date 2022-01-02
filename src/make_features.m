function [] = make_features(config)
    path=genpath('library');
    addpath(path);
    if ~exist('config','var')
        config = configuration;
        config = config.updateModelLocation();
        config.split=15;
    end
    data_dir = config.splited_dir;
    split = config.split;

    datasets = {'train', 'val', 'test'}    
    system(['rm -rf '...
    config.features_dir,...
    '/',num2str(config.feature),'/*',...
    ])
    for h = 1:length(datasets)
        %get all the subjects in the splitted directory
        dataset_dir = [data_dir,'/',num2str(split),'/',datasets{h}];
        subjects = get_all_sub_dir(dataset_dir);
        for i = 1:length(subjects)
            %get all the session for a given subject
            disp(['Extracting features for subject ',num2str(i),'/',num2str(length(subjects))])
            session = get_all_sub_dir([...
                dataset_dir,...
                '/',subjects{i}
                ]);
            for j = 1:length(session)
                session_dir=[...
                dataset_dir,...
                '/',subjects{i},...
                '/',session{j}
                ];
                experiments = get_all_sub_dir(session_dir);
                for k = 1:length(experiments)
                    experiments_dir = [session_dir,'/',...
                                    experiments{k}];
                    recordings = dir([experiments_dir,...
                        '/*.mat'
                        ]);
                    recordings = {recordings(:).name};
                    save_dir=[...
                        config.features_dir,...
                        '/',num2str(config.split),...
                        '/',num2str(config.feature),...
                        '/',datasets{h},...
                        '/',subjects{i},...
                        '/',session{j},...
                        '/',experiments{k},...
                        '/'
                        ];
                    mkdir(save_dir);
                    for l = 1:length(recordings)
                        if isfile([save_dir,...
                            '/',recordings{l}])
                            continue;
                        end
                        data = load([...
                            experiments_dir,...
                            '/',recordings{l}
                            ]);
                        freq = data.sampling_rate;
                        data = data.data;
                        if config.dataset ==1
                            channels =  config.lobe_map{config.lobe};
                            data = data(channels,:);
                        end;
                        feature = config.feature_function{config.feature}(config,data, config.samp_rate);
                        SaveFeatures([...
                            save_dir,...
                            '/',recordings{l}],feature);
                        
                    end
                end
            end
        end
    end
    %poolobj = gcp('nocreate');
    %delete(poolobj);
    disp('done')
end

function []= SaveFeatures(dir,feature)
    save(dir,'feature')
end

