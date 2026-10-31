function [S, F, T] = spectrogram(x, window, noverlap, f, fs)
% Octave shim for MATLAB spectrogram(x, window, noverlap, f, fs) where f is
% a vector of frequencies in Hz (MATLAB evaluates the DFT at exactly those
% frequencies via the Goertzel algorithm). Only the call form used by this
% repository is supported. An integer window means a Hamming window of that
% length (MATLAB default).
  x = x(:).';
  if isscalar(window)
    win = hamming(window).';
  else
    win = window(:).';
  end
  nwin = numel(win);
  if nargin < 3 || isempty(noverlap)
    noverlap = floor(nwin/2);
  end
  if nargin < 5 || isempty(fs)
    fs = 2*pi;  % normalised frequency
  end
  if nargin < 4 || isempty(f) || isscalar(f)
    error('octave_compat/spectrogram: only the frequency-vector form is implemented');
  end
  hop = nwin - noverlap;
  ncol = fix((numel(x) - noverlap) / hop);
  idx = (1:nwin).' + (0:ncol-1) * hop;          % nwin x ncol
  frames = x(idx) .* win.';                      % windowed segments
  F = f(:);
  n = 0:nwin-1;
  E = exp(-1i * 2 * pi * F * n / fs);           % nf x nwin DFT basis
  S = E * frames;                                % nf x ncol
  T = ((0:ncol-1) * hop + nwin/2) / fs;
end
