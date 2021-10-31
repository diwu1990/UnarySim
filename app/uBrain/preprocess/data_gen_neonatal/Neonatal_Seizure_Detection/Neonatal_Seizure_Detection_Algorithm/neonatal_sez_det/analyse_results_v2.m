function  [td, tdl, td_ref, td_per, fd, fdur, sez_events, auto_events, N, sens, spec, Npos, Nneg] = analyse_results_v2(ah, ar)
% 
% This function estimates the sensitivty, the number of events correctly
% determined and the number of false detections
%
% Inputs: ah is the human annotation
%                ar is the automated detections
%
% Outputs: td is the number of seizures correctly detected
%                   tdl is the duration of the detected seizure
%                   tdref is the seizure that was detected
%                   tdper is duration of the seizure detected
%                   fd is the number of false detections
%                   fdur is the duration of the false detections
%                   sez_events is the number of seizures in the human annotation
%                   auto_events is the number of seizures in the automated
%                   annoations
%                  N is the length of the annotations
%                  sens is the sensitivity
%                 spec is the specidificity
%                 Npos is the duration of all seizure (seizure burden)
%                 Nneg is the duration of all nonseizure (seizure burden)
%
% Dependencies: detections_v2.m
%
% Nathan Stevenson
%

% EVENT BASED ANALYSIS
ah = [0 ah 0];
ar = [0 ar 0];
N = length(ah);
ref1 = find(diff(ah)==1);
ref2 = find(diff(ah)==-1);
reh = [ref1+1 ; ref2];
Q1 = size(reh);
sez_events = Q1(2);
clear ref1 ref2
ref1 = find(diff(ar)==1);
ref2 = find(diff(ar)==-1);
rer = [ref1+1 ; ref2];
Q2 = size(rer);
auto_events = Q2(2);

clear ref1 ref2

if sez_events == 0 
    td = 0; tdl = []; td_ref = 0;
    fd = auto_events;
    fdur = rer(2,:)-rer(1,:); td_per=0;
else
    if auto_events == 0
        td = 0;
        fd = 0; fdur = [];
        tdl = []; td_ref = 0; td_per = 0;
    else
    [td, tdl, td_ref, td_per, fd, fdur] = detections_v2(reh, rer, ah, ar);
    end
end

% SECOND BY SECOND ANALYSIS
ah = ah(2:end-1);
ar = ar(2:end-1);
N = length(ar);
ref1 = find(ah==1);
Npos = length(ref1);
if isempty(Npos)==1
    sens = 0;
    Npos = 0;
else
sens = length(find(ar(ref1)==1))/Npos;
end
ref2 = find(ah==0);
Nneg = length(ref2);
if isempty(Nneg)==1
   spec = 0;
   Nneg = 0;
else
spec = length(find(ar(ref2)==0))/Nneg;
end

