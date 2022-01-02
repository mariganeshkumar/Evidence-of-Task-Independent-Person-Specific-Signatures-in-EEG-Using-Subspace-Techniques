function sub_dirs = get_all_sub_dir(dir_path)
    sub_dirs = dir(dir_path);
    is_dir = [sub_dirs(:).isdir];
    sub_dirs = {sub_dirs(is_dir).name};
    sub_dirs(ismember(sub_dirs,{'.','..'})) = [];
end