function [number_of_patients,directorynames]=getNroOfPatients(path,file_type)
%file_type - type of the file to be searched. e.g '.mat' or '.edf', ...
files = dir([path, '/*' file_type]);
directoryNames = {files.name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..'}));
[directorynames] = sort_nat(directoryNames);
number_of_patients=length(directorynames);