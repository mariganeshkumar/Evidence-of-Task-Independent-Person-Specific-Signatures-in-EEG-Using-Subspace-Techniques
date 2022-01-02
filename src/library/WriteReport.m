function [] = WriteReport(result_str, accuracy, predicted_label, gt_label, eer, norm_eer, subjectsInfo)
num_subjects=length(unique(gt_label));
cm = confusionmat(predicted_label,gt_label);
cm
missclassification = zeros(1, num_subjects);
for i=1:num_subjects
    missclassification(i)=(sum(cm(i,:))-cm(i,i))/sum(cm(i,:))*100;
end

fId = fopen(result_str,'w');
strr = '~~~~Classification Report~~~~\n';
fprintf(fId,strr);
disp(strr);
[sortedMC,sortInd]=sort(missclassification,'descend');
for i = 1:length(sortInd)
    if sortedMC(i) > 0
        strr = strcat('Subject ',subjectsInfo{sortInd(i)},' miss classified ', num2str(sortedMC(i)),'%% of trails miss classified\n');
        fprintf(fId,strr);
        disp(strr);
        missClassifiedClasses=cm(sortInd(i),:);
        [sortedMCC,sortIndMCC]=sort(missClassifiedClasses,'descend');
        mccReport='';
        for j = 1:length(sortIndMCC)
            if sortIndMCC(j)~=sortInd(i) && sortedMCC(j) > 0
                mccReport = strcat(mccReport,'>>>',...
                    subjectsInfo{sortIndMCC(j)},...
                    ': ',...
                    num2str(sortedMCC(j)/sum(cm(sortInd(i),:))*100),...
                    '%%<<<');
            end
        end
        
        strr = strcat('    miss classified with :',mccReport,' \n');
        fprintf(fId,strr);
        disp(strr);
    end
end

strr ='~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~';
fprintf(fId,strr);
disp(strr)


strr = ['No Of Subjects:', num2str(length(subjectsInfo)), '\n'];
fprintf(fId,strr);
disp(strr);

strr = 'Top 10 miss classified classes: ';
for i = 1:length(sortInd)
    strr = [strr, subjectsInfo{sortInd(i)}, ' '];
end
fprintf(fId,strr);
disp(strr);

strr = ['accuracies  = ',num2str(accuracy),'\n'];
fprintf(fId,strr);
disp(strr);

strr = ['mean  = ',num2str(mean(accuracy)),'  std  = ', num2str(std(accuracy))];
fprintf(fId,strr);
disp(strr);

strr = ['EERs  = ',num2str(eer),'\n'];
fprintf(fId,strr);
disp(strr);

strr = ['mean  = ',num2str(mean(eer)),'  std  = ', num2str(std(eer))];
fprintf(fId,strr);
disp(strr);


strr = ['Normalised EERs  = ',num2str(norm_eer),'\n'];
fprintf(fId,strr);
disp(strr);

strr = ['mean  = ',num2str(mean(norm_eer)),'  std  = ', num2str(std(norm_eer))];
fprintf(fId,strr);
disp(strr);

fclose(fId);
end

