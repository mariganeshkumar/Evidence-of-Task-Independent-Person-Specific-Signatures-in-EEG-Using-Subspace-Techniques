% Compute EERs with the bundled BOSARIS toolkit, exactly as plot_score_distribution.m:
%   [Pmiss, Pfa] = rocch(target_scores, non_target_scores); EER = rocch2eer(Pmiss, Pfa) * 100
% Input  RT_SCORES (.mat): pairs <name>_t (target scores) and <name>_n (non-target scores).
% Output RT_EER_OUT (text): one line "<name> <EER in %>" per pair.
% Run from src/ (called by tools/reproduce_tables.sh).
more off; page_screen_output(false);
addpath(genpath(fullfile('library', 'bosaris_toolkit')));
in = getenv('RT_SCORES'); out = getenv('RT_EER_OUT');
if isempty(in) || isempty(out), error('RT_SCORES and RT_EER_OUT must be set'); end
d = load(in);
k = fieldnames(d);
fid = fopen(out, 'w');
for i = 1:numel(k)
  nm = k{i};
  if numel(nm) > 2 && strcmp(nm(end-1:end), '_t')
    b = nm(1:end-2);
    [Pmiss, Pfa] = rocch(double(d.([b '_t'])), double(d.([b '_n'])));
    fprintf(fid, '%s %.10f\n', b, rocch2eer(Pmiss, Pfa) * 100);
  end
end
fclose(fid);
