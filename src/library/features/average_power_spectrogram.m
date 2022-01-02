function [output] = average_power_spectrogram(config,data,samp_rate)
    output = compute_average_power_spectrogram(...
                                               data,samp_rate,...
                                               round(config.win_size/1000*samp_rate),...
                                               round(config.overlap/1000*samp_rate),...
                                               config.lfreq,...
                                               config.hfreq,...
                                               config.nfft...
                                               );
end


function [output] = compute_average_power_spectrogram(input_mat, sampling_rate, winSize, overlap, low, high, nFFT)
% 
% input_mat = A matrix having No of Channels*time
% winSize = Window size (number of frame) for spectrum
% overlap = Overlap size (number of frame) to calculate spectrum
% low = lower frequency of the band for which average spectrum is to be calculated
% high = higher frequency of the band for which average spectrum is to be calculated
% sampling_rate = Sampling rate of the EEG signal

% OUTPUT
% output = num of channels*freq*timeFrame 

 if ~exist('nFFT','var')
    a = winSize;
    count  = 1;
    while a > 1
        count  = count + 1;
        a = floor(a/2);
    end
    nFFT = 2^count;
 end


[m, ~] = size(input_mat);
input_mat =  double(input_mat);
aps = [];
for i=1:m
    [s,f,~] = spectrogram(input_mat(i,:), winSize, overlap,[low:(sampling_rate/nFFT):high], sampling_rate);
    s = abs(s);
    aps(i,:,:)  = s;
end
output = aps;
end

