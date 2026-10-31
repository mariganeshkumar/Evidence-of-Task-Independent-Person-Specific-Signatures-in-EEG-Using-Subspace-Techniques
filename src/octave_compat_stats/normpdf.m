function y = normpdf(x, mu, sigma)
% Minimal Octave shim for normpdf
  if nargin < 2, mu = 0; end
  if nargin < 3, sigma = 1; end
  y = exp(-0.5 * ((x - mu) ./ sigma).^2) ./ (sqrt(2*pi) .* sigma);
end
