function recordings = find_mat_files_recursive(root)
% Octave replacement for dir([root,'/**/*.mat']): returns a struct array
% with the same fields as dir(), searched recursively.
  recordings = dir(fullfile(root, '*.mat'));
  sub = dir(root);
  sub = sub([sub.isdir] & ~ismember({sub.name}, {'.', '..'}));
  for i = 1:numel(sub)
    r = find_mat_files_recursive(fullfile(root, sub(i).name));
    recordings = [recordings; r(:)];
  end
end
