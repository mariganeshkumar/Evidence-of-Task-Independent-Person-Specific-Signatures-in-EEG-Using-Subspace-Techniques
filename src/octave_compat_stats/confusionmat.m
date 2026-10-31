function [C, order] = confusionmat(g1, g2)
% Minimal Octave shim for MATLAB confusionmat(known, predicted) with numeric
% labels: rows = known (g1), cols = predicted (g2), order = sorted union.
  g1 = g1(:); g2 = g2(:);
  order = unique([g1; g2]);
  [~, i1] = ismember(g1, order);
  [~, i2] = ismember(g2, order);
  C = accumarray([i1 i2], 1, [numel(order) numel(order)]);
end
