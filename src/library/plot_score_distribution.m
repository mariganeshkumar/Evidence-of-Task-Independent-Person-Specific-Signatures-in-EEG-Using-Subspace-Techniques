function [eer] = plot_score_distribution(save_file, target, non_target)
mean1=mean(target);
mean2=mean(non_target);
std1=std(target);
std2=std(non_target);
target = reshape(target, [1 length(target)]);
non_target = reshape(non_target, [1 length(non_target)]);
score = [target,non_target];
min_score = mean2-3*std2;
max_score = mean1+3*std1;
inc = (max_score -min_score)/2000;
values = min_score:inc:max_score;
h=figure();
dist1 = normpdf(values,mean1,std1);
dist2 = normpdf(values,mean2,std2);
area(values, dist1, 'FaceAlpha', 0.3, 'FaceColor','red', 'EdgeColor','red', 'LineWidth', 2)
hold on;
area(values,dist2, 'FaceAlpha', 0.3, 'FaceColor','blue', 'EdgeColor','blue', 'LineWidth', 2)
div=kld(inc,dist1,dist2);

[Pmiss,Pfa] = rocch(target, non_target);
eer = rocch2eer(Pmiss,Pfa) * 100;
title(['KLD: ', num2str(div), ' EER: ',num2str(eer)]);
saveas(h,[save_file,'.png']);
saveas(h,[save_file,'.fig']);
saveas(h,[save_file,'.svg']);

end


function div = kld(dx,pVect1,pVect2)
KL1 = dx*sum(pVect1 .* (log(pVect1)-log(pVect2)));
KL2 = dx*sum(pVect2 .* (log(pVect2)-log(pVect1)));
div = (KL1+KL2)/2;
end