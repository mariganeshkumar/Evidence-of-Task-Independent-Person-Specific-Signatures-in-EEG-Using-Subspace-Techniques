% Chunk and featurise Rini's files for the 5 recordings used in Tables 2-3.
%
% Rini's files are unprocessed Net Station exports of the same sessions as five
% held-out Passive-Audio test recordings of Dataset 1. They are not distributed
% with this repository.
%
% The chunking is the logic of make_data.m (divIntoValTestChunks) and the
% features are computed with make_features.m's feature function, using the same
% src/configuration.m, so the only difference from the pipeline is the input file.
%
% NOTE: Rini's files are fed WITHOUT amplitude rescaling: the raw Net Station
% values go straight into chunking/feature extraction, exactly as the pipeline
% treats the preprocessed dataset files. All 129 channels are kept in Net Station
% order, so the 9-channel montage indices of configuration.m apply unchanged.
%
% Called by tools/reproduce_tables.sh (run from src/) with environment variables:
%   RT_RINI_DIR  directory with <subject>_timit_hear_only_<date>_<time>.mat
%   RT_OUT_DIR   output directory; features go to RT_OUT_DIR/<subject>/<i><time>.tmp.mat
more off; page_screen_output(false); save_default_options('-v7');
addpath(fullfile(pwd, 'octave_compat'));
addpath(genpath('library'));
rini_dir = getenv('RT_RINI_DIR'); out_base = getenv('RT_OUT_DIR');
if isempty(rini_dir) || isempty(out_base), error('RT_RINI_DIR and RT_OUT_DIR must be set'); end
config = configuration;
chans = config.lobe_map{config.lobe};
split = config.split; per = config.val_per;
fs_expected = 250;
recs = {'012100073','20180819','040327'; '014100078','20180822','043400'; '015100076','20180822','080342'; ...
        '016100071','20180819','025325'; '017000070','20180822','010640'};
for r = 1:size(recs,1)
  s = recs{r,1}; d = recs{r,2}; t = recs{r,3};
  f = sprintf('%s/%s_timit_hear_only_%s_%s.mat', rini_dir, s, d, t);
  if ~exist(f, 'file'), error('missing Rini file: %s', f); end
  S = load(f);
  fn = fieldnames(S); k = fn(~cellfun(@isempty, regexp(fn, '^a.*mff$')));
  if numel(k) ~= 1, error('%s: expected exactly one variable named a...mff', f); end
  all_chan_data = S.(k{1});   % used as stored: no amplitude rescaling (see NOTE above)
  if S.EEGSamplingRate ~= fs_expected, error('%s: unexpected sampling rate %g', f, S.EEGSamplingRate); end
  % ---- same logic as divIntoValTestChunks in make_data.m ----
  current_data = all_chan_data(:, 250*5:end-250*5);
  current_data = current_data - mean(current_data, 2);
  tot_chunks = floor(size(current_data,2)/(split*250));
  val_chunks = floor(tot_chunks*per);
  printf('  %s %s_%s: %d x %d samples, %d test chunks\n', s, d, t, rows(all_chan_data), columns(all_chan_data), tot_chunks-val_chunks);
  sd = sprintf('%s/%s', out_base, s);
  if ~exist(sd, 'dir'), mkdir(sd); end
  for i = val_chunks:tot_chunks-1        % test chunks only
    data = current_data(:, 1+i*250*split:(i+1)*250*split);
    data = data - mean(data, 2);
    % ---- same as make_features.m: select the montage channels, apply the feature function ----
    x = data(chans, :);
    feature = config.feature_function{config.feature}(config, x, config.samp_rate);
    save([sd '/' num2str(i) t '.tmp.mat'], 'feature');
  end
end
disp('  Rini features done');
