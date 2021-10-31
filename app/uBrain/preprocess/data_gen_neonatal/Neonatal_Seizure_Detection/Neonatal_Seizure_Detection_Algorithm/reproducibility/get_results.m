% Final results 
function results=get_results(group,annotat,dec)
% This function estimates a raft of performance measures comparing the SVM
% output in variable dec, with the annotations of the human expert in
% variable annotat_new. Variable cn is the optimal length of the collaring
% extension

%  auc - is the auc on a per baby basis 1x79 with NaN when no seizure is present in a baby
%  auc1 - is the auc of all recording concatenated together 1x1
%  acc - is the accuracy over a range of 1000 thresholds per baby 1000x79
%  tdr1 - the seizure detection rate on a per baby basis with 1000 thresholds 1000x79
%  fdr1 - the false detection rate per hour on a baby per baby basis with 1000 thresholds 1000x79
%  tdr2 - the seizure detection rate over the concatenated recordings with 1000 thresholds 1x1000
%  fdr2 - the false detections per hour over the concantenated recordings with 1000 thresholds 1x1000
%  kap - the kappa value between human experts with a 95%CI 1x3
%  dkap - the 95%CI change in kappa when replacing a human expert with the SDA 3x2
%  ddkap - the kappa when replacing a human expert with the SDA 1x3
%  dist - differences to human expert
%  auc_boot - aucs on bootstrapped, concatenated data to define
%  significant differences between algorithms
%  vva, vvb and vvc - diffferences to human experts with descending number of patients
%  After the difference becomes negative, the benchmark is reached
%  sens - sensitivity
%  spec - specificity
%  ppv - positive predictive value
%  npv - negative predictive value
%  auc_th - auc pre patient on single threshold

% Number of thresholds and iterations
num=1000;

% Original DB algorithm
if isequal(group,'SDA_DB')
    cn=28;
    [auc, auc1,auc_boot, tdr1, fdr1, tdr2, fdr2, kap, dkap,ddkap,dist,sens,spec,ppv,npv] = compute_results_DB_orig(dec, annotat, cn,num);
    [vva, vvb, vvc,~] = descend_agree_DBorig(dec, annotat, cn);
    
    results.auc_concat=auc1; results.auc=auc; results.auc_boot=auc_boot;
    results.tdr=tdr1; results.fdr=fdr1; results.tdr_concat=tdr2; results.fdr_concat=fdr2; 
    results.kap_human=kap; results.dkap95CI=dkap; 
    results.kap_replaced=ddkap; results.dkap_dist=dist;
    results.dkap_descend=vvb; results.sens=sens; results.spec=spec;
    results.ppv=ppv; results.npv=npv; 
else
    % Proposed SDA
    if isequal(group,'SDA')
        cn=23;
    % Modified DB    
    elseif isequal(group,'SDA_DB_mod')
        cn=28;
    % Temko    
    elseif isequal(group,'SDA_T')
        cn=7; 
    end
    %[auc, auc1, acc, tdr1, fdr1, tdr2, fdr2, kap, dkap,ddkap,dist] = compute_results(dec, annotat_new, cn);
    [auc, auc1, acc, tdr1, fdr1, tdr2, fdr2, kap, dkap,ddkap,dist,thr1] = compute_results_NJS(dec, annotat, cn);

    [auc_boot,~] = auc_ci_concat(dec, annotat, cn,num);
    [vva, vvb, vvc,~] = descend_agree(dec, annotat, cn);
    [sens, spec, ppv, npv,auc_th] = single_th_results(dec, annotat, cn);

    results.auc_concat=auc1; results.auc=auc; results.acc=acc; results.auc_boot=auc_boot;
    results.tdr=tdr1; results.fdr=fdr1; results.tdr_concat=tdr2; results.fdr_concat=fdr2; 
    results.kap_human=kap; results.dkap95CI=dkap; 
    results.kap_replaced=ddkap; results.dkap_dist=dist;
    results.dkap_descend=vvb; results.sens=sens; results.spec=spec;
    results.ppv=ppv; results.npv=npv; results.auc_single_th=auc_th;
    results.opt_th=thr1;
end
