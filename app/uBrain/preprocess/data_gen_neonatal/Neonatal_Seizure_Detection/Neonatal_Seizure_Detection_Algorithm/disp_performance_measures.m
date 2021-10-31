% Results for IJNS
function disp_performance_measures(path_SDA,path_SDA_DB,path_SDA_DB_mod,path_SDA_T)
%
load(path_SDA);
auc1=results.auc; auc_cc1=results.auc_concat;
auc_boot1=results.auc_boot;
sens1=results.sens;spec1=results.spec;ppv1=results.ppv;npv1=results.npv;
auc_th1=results.auc_single_th;
tdr11=results.tdr;fdr11=results.fdr;
dk_dk95_1=cat(2,mean(results.dkap_dist,2),results.dkap95CI);
mean_k1=mean(results.kap_replaced);
descend=results.dkap_descend;
[num1, pp1]=find_num_benchmark(descend);
load(path_SDA_DB_mod);
auc2=results.auc; auc_cc2=results.auc_concat;
auc_boot2=results.auc_boot;
sens2=results.sens;spec2=results.spec;ppv2=results.ppv;npv2=results.npv;
auc_th2=results.auc_single_th;
tdr12=results.tdr;fdr12=results.fdr;
dk_dk95_2=cat(2,mean(results.dkap_dist,2),results.dkap95CI);
mean_k2=mean(results.kap_replaced);
descend=results.dkap_descend;
[num2, pp2]=find_num_benchmark(descend);
load(path_SDA_T);
auc3=results.auc; auc_cc3=results.auc_concat;
auc_boot3=results.auc_boot;
sens3=results.sens;spec3=results.spec;ppv3=results.ppv;npv3=results.npv;
auc_th3=results.auc_single_th;
tdr13=results.tdr;fdr13=results.fdr;
dk_dk95_3=cat(2,mean(results.dkap_dist,2),results.dkap95CI);
mean_k3=mean(results.kap_replaced);
descend=results.dkap_descend;
[num3, pp3]=find_num_benchmark(descend);
load(path_SDA_DB);
auc4=results.auc; auc_cc4=results.auc_concat;
auc_boot4=results.auc_boot;
sens4=results.sens;spec4=results.spec;ppv4=results.ppv;npv4=results.npv;
tdr14=results.tdr;fdr14=results.fdr;
dk_dk95_4=cat(2,mean(results.dkap_dist,2),results.dkap95CI);
mean_k4=mean(results.kap_replaced);
descend=results.dkap_descend;
[num4, pp4]=find_num_benchmark(descend);

% AUCs
aucs=[mean(auc1(isnan(auc1)==0)),median(auc1(isnan(auc1)==0)),quantile(auc1(isnan(auc1)==0),[0.25,0.75]);...
    mean(auc2(isnan(auc2)==0)),median(auc2(isnan(auc2)==0)),quantile(auc2(isnan(auc2)==0),[0.25,0.75]);...
    mean(auc3(isnan(auc3)==0)),median(auc3(isnan(auc3)==0)),quantile(auc3(isnan(auc3)==0),[0.25,0.75]);...
    mean(auc4(auc4~=0)),median(auc4(auc4~=0)),quantile(auc4(auc4~=0),[0.25 0.75])];
%
disp('Table 3:')

fprintf(['\n\nMedian AUC (IQR) Proposed SDA: %.3f(%.3f-%.3f) \n Median AUC (IQR) SDA_DB: %.3f(%.3f-%.3f) ',...
'\n Median AUC (IQR) SDA_mDB: %.3f(%.3f-%.3f) \n Median AUC (IQR) SDA_T: %.3f(%.3f-%.3f) '...
'\n Mean AUC Proposed SDA: %.3f \n Mean AUC SDA_DB: %.3f\n Mean AUC SDA_mDB: %.3f',...
'\n Mean AUC SDA_T: %.3f \n'],...
aucs(1,2:4),aucs(4,2:4),aucs(2,2:4),aucs(3,2:4),aucs(1,1),aucs(4,1),aucs(2,1),aucs(3,1));

% P-values
% p-values  between algorithms
p_values=[0,signrank(auc1,auc2),signrank(auc1,auc3),signrank(auc1,auc4);...
    0,0,signrank(auc2,auc3),signrank(auc2,auc4);...
    0,0,0,signrank(auc3,auc4);...
    0,0,0,0];

% Delta AUCs 95% CIs
deltaAUC95=[quantile(abs(auc_boot1-auc_boot2),[0.025 0.975]);...
quantile(abs(auc_boot1-auc_boot3),[0.025 0.975]);...
quantile(abs(auc_boot1-auc_boot4),[0.025 0.975])];

% TDR

% SDA
r1 = find(auc1>0);
for ii = 1:length(r1)
    a = tdr11(:,r1(ii));
    b = fdr11(:,r1(ii));
    zz1 = find(b<1);xx1=find(b==0);
    zz2 = find(zz1 > find(b==max(b),1),1);
    xx2 = find(xx1 > find(b==max(b),1),1);
    if isempty(zz2)==1
        tt2(ii)=1;
    else
        tt2(ii) = a(zz1(zz2));
    end
    if isempty(xx2)==1
        tt3(ii)=1;
    else
        tt3(ii) = a(xx1(xx2));
    end
end
sdr_sda=[mean(tt2),median(tt2), quantile(tt2,[0.25, 0.75]);...
    mean(tt3),median(tt3), quantile(tt3,[0.25, 0.75])];
% DB_mod
for ii = 1:length(r1)
    a = tdr12(:,r1(ii));
    b = fdr12(:,r1(ii));
    zz1 = find(b<1);xx1=find(b==0);
    zz2 = find(zz1 > find(b==max(b),1),1);
    xx2 = find(xx1 > find(b==max(b),1),1);
    if isempty(zz2)==1
        tt2(ii)=1;
    else
        tt2(ii) = a(zz1(zz2));
    end
    if isempty(xx2)==1
        tt3(ii)=1;
    else
        tt3(ii) = a(xx1(xx2));
    end
end

sdr_db_mod=[mean(tt2),median(tt2), quantile(tt2,[0.25, 0.75]);...
    mean(tt3),median(tt3), quantile(tt3,[0.25, 0.75])];
% Temko
for ii = 1:length(r1)
    a = tdr13(:,r1(ii));
    b = fdr13(:,r1(ii));
    zz1 = find(b<1);xx1=find(b==0);
    zz2 = find(zz1 > find(b==max(b),1),1);
    xx2 = find(xx1 > find(b==max(b),1),1);
    if isempty(zz2)==1
        tt2(ii)=1;
    else
        tt2(ii) = a(zz1(zz2));
    end
    if isempty(xx2)==1
        tt3(ii)=1;
    else
        tt3(ii) = a(xx1(xx2));
    end
end
sdr_t=[mean(tt2),median(tt2), quantile(tt2,[0.25, 0.75]);...
    mean(tt3),median(tt3), quantile(tt3,[0.25, 0.75])];
% DB original
for ii = 1:length(r1)
    a = tdr14(:,r1(ii));
    b = fdr14(:,r1(ii));
    zz1 = find(b<1);xx1=find(b==0);
    zz2 = find(zz1 > find(b==max(b),1),1);
    xx2 = find(xx1 > find(b==max(b),1),1);
    if isempty(zz2)==1
        tt2(ii)=1;
    else
        tt2(ii) = a(zz1(zz2));
    end
    if isempty(xx2)==1
        tt3(ii)=1;
    else
        tt3(ii) = a(xx1(xx2));
    end
end
sdr_db_orig=[mean(tdr14(r1)),median(tdr14(r1)), quantile(tdr14(r1),[0.25,0.75])];
fdr_db_orig=[mean(fdr14),median(fdr14), quantile(fdr14,[0.25,0.75])];
%
fprintf(['\n Mean SDR (FD/h~0) Proposed SDA %2.1f ',...
    '\n Mean SDR (FD/h~0) SDA_mDB %2.1f ',...
    '\n Mean SDR (FD/h~0) SDA_T %2.1f ',...
    '\n Mean SDR (FD/h~1) Proposed SDA %2.1f ',...
    '\n Mean SDR (FD/h~1) SDA_mDB %2.1f ',...
    '\n Mean SDR (FD/h~1) SDA_T %2.1f \n'],sdr_sda(2,1)*100,sdr_db_mod(2,1)*100,sdr_t(2,1)*100,...
    sdr_sda(1,1)*100,sdr_db_mod(1,1)*100,sdr_t(1,1)*100);

disp('p-values (SDA_DB,SDA_mDB,SDA_T):');disp(p_values(1,2:4))

fprintf(['\n deltaAUC 95CI SDA_DB %.3f-%.3f ',...
    '\n deltaAUC 95CI SDA_mDB %.3f-%.3f ',...
    '\n deltaAUC 95CI SDA_T %.3f-%.3f \n'],...
    deltaAUC95(3,:),deltaAUC95(1,:),deltaAUC95(2,:));

fprintf(['\n\n Median SDR (IQR) SDA_DB %2.1f (%2.1f-%2.1f)',...
    '\n Median FD/h %.3f (%.3f-%.3f)\n'],sdr_db_orig(2:4)*100,fdr_db_orig(2:4));
% Sens spec npv
ss=r1;

% Single th AUC
aucs_single_th=[median(auc_th1(ss)),quantile(auc_th1(ss),[0.25,0.75]);...
    median(auc_th2(ss)),quantile(auc_th2(ss),[0.25,0.75]);...
    median(auc_th3(ss)),quantile(auc_th3(ss),[0.25,0.75])];

%
fprintf(['\n\nMedian(IQR) single threshold AUC SDA %.3f(%.3f-%.3f)',...
    '\nMedian(IQR) single threshold AUC SDA_mDB %.3f(%.3f-%.3f)',...
    '\nMedian(IQR) single threshold AUC SDA_T %.3f(%.3f-%.3f) \n'],...
    aucs_single_th(1,:),aucs_single_th(2,:),aucs_single_th(3,:));
%

sn_sp=[median(sens1(ss)),quantile(sens1(ss),[0.25,0.75]);...
    median(spec1),quantile(spec1,[0.25,0.75])];%...
%     median(sens2(ss)),quantile(sens2(ss),[0.25,0.75]);...
%     median(spec2),quantile(spec2,[0.25,0.75]);...
%     median(sens3(ss)),quantile(sens3(ss),[0.25,0.75]);...
%     median(spec3),quantile(spec3,[0.25,0.75]);...
%     median(sens4(ss)),quantile(sens4(ss),[0.25,0.75]);...
%     median(spec4),quantile(spec4,[0.25,0.75])];

ppv_npv=[median(ppv1(ss)),quantile(ppv1(ss),[0.25,0.75]);...
    median(npv1),quantile(npv1,[0.25,0.75])];%...
%     median(ppv2(ss)),quantile(ppv2(ss),[0.25,0.75]);...
%     median(npv2),quantile(npv2,[0.25,0.75]);...
%     median(ppv3(ss)),quantile(ppv3(ss),[0.25,0.75]);...
%     median(npv3),quantile(npv3,[0.25,0.75]);...
%     median(ppv4(ss)),quantile(ppv4(ss),[0.25,0.75]);...
%     median(npv4),quantile(npv4,[0.25,0.75]);...
%     ];
% Sens spec
fprintf(['\n\nMedian(IQR) sensitivity %.3f(%.3f-%.3f)',...
    '\nMedian(IQR) specificity %.3f(%.3f-%.3f)',...
    '\nMedian(IQR) positive predictive value %.3f(%.3f-%.3f)',...
    '\nMedian(IQR) negative predictive value %.3f(%.3f-%.3f)\n'],...
    sn_sp(1,:),sn_sp(2,:),ppv_npv(1,:),ppv_npv(2,:));

% Print table 4
fprintf(['\n\n Table 4:',...
    '\ndk(95CI) Proposed SDA',... 
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\ndk(95CI) SDA_DB',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\ndk(95CI) SDA_mDB,',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\ndk(95CI) SDA_T',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)',...
    '\n %.3f(%.3f-%.3f)\n'],...
    dk_dk95_1(1,:),dk_dk95_1(2,:),dk_dk95_1(3,:),...
    dk_dk95_4(1,:),dk_dk95_4(2,:),dk_dk95_4(3,:),...
    dk_dk95_2(1,:),dk_dk95_2(2,:),dk_dk95_2(3,:),...
    dk_dk95_3(1,:),dk_dk95_3(2,:),dk_dk95_3(3,:));

fprintf(['\n\n Mean k %.3f %.3f %.3f %.3f\n'],mean_k1,mean_k4,mean_k2,mean_k3);

fprintf(['\n\n Cohort size %2.f(%2.f) %2.f(%2.f) %2.f(%2.f) %2.f(%2.f)\n'],...
    num1,pp1,num4,pp4,num2,pp2,num3,pp3);
