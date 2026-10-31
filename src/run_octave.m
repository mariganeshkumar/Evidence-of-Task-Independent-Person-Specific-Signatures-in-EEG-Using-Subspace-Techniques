% GNU Octave entry point: sets up Octave compatibility, then runs run.m.
% Usage (from src/):  octave-cli run_octave.m
% MATLAB users should keep using run.m; the shims in octave_compat*/ are only
% added to the path here, and only when running under Octave, so they never
% shadow MATLAB built-ins (spectrogram, confusionmat, normpdf).
if ~exist('OCTAVE_VERSION', 'builtin')
  error('run_octave.m is for GNU Octave only; in MATLAB use run.m');
end
more off;
page_screen_output(false);
save_default_options('-v7');          % MATLAB-readable binary .mat (Octave default is text)
set(0, 'defaultfigurevisible', 'off');
addpath(fullfile(pwd, 'octave_compat'));          % spectrogram(x,win,ovl,fvec,fs)
pkgs = pkg('list'); pkgs = cellfun(@(p) p.name, pkgs, 'UniformOutput', false);
if any(strcmp(pkgs, 'statistics'))
  pkg load statistics;
else
  addpath(fullfile(pwd, 'octave_compat_stats'));  % confusionmat, normpdf
end
source('run.m');
