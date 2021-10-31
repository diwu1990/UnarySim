function dat = preprocess(dat_eeg)
% This function preprocess EEG data with a bandpass and notch filter

load filters_db_256
dat = filter(Bn, An, dat_eeg); % 50Hz notch filter
dat = filter(NumL, DenL, dat); % Low pass IIR Butterworth, cutoff 30Hz
dat = filter(NumH, DenH, dat); % High pass IIR Butterwoth, cutoff 0.3Hz


