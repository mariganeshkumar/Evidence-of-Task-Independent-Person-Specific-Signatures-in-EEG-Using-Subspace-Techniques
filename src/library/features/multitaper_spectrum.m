function [feature] =  multitaper_spectrum(config,data,samp_rate)
feature = compute_multitaper_spectrum(...
                                       data,samp_rate,...
                                       config.win_size/1000*samp_rate,...
                                       config.overlap/1000*samp_rate,...
                                       config.lfreq,...
                                       config.hfreq...
                                       );

end

function [features] = compute_multitaper_spectrum(data, sampling_rate, winSize, overlap, low, high)
movingwin=[winSize/sampling_rate (winSize-overlap)/sampling_rate];
params.Fs=sampling_rate;
params.fpass=[low high];
params.tapers=[5 9]; %todo: make this parameter configurable
params.pad = 0;

[S_d,~,~] = mtspecgramc(data', movingwin, params);
%S_d = S_d-repmat(mean(S_b),size(S_d,1),1);
features = permute(S_d,[3, 2 ,1 ]);

end