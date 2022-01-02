config = configuration()

system(['rm -rf ',' ',config.tmp_dir])

if ~exist([config.splited_dir,'/',num2str(config.split)], 'dir')
    make_data(config)
end
