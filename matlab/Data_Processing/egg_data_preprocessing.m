clear all
clc
close all

path = "C:\Personal_Folder\Vocal_CQ\EGG_data\test";
wav_list = dir(path);
wav_list = {wav_list.name};
wav_list(1:2) = [];
n = length(wav_list);

list_CQ = [];
for i = 1:n
    wav_path = strcat(path, '\\', string(wav_list(i)));
    list_CQ = [list_CQ, CG_evaluate(wav_path, 50, 0.4)];
end

fileID = fopen('wav_CQ_test.txt', 'w');
for i = 1:length(wav_list)
    fprintf(fileID, '%s, %.10f\n', wav_list{i}, list_CQ(i));
end

fclose(fileID);