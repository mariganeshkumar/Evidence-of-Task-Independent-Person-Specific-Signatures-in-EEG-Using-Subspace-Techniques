config = configuration()

system(['rm -rf ',' ',config.tmp_dir])

if ~exist([config.splited_dir,'/',num2str(config.split)], 'dir')
    make_data(config)
end

if ~exist([config.features_dir,'/',num2str(config.split)], 'dir')
    make_features(config)
end

train_and_test_eeg_biometrics(config)