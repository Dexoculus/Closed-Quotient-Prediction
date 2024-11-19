import os

folder_path = 'C:\\Personal_Folder\\EGG_data_cut'
folder_list = os.listdir(folder_path)
new_folder_path = 'C:\\Personal_Folder\\EGG_data_renamed'

for i, name in enumerate(folder_list):
    inner_folder = f'C:\\Personal_Folder\\EGG_data_cut\\{name}'
    name_element = name.split('_')
    wav_list = os.listdir(inner_folder)
    for j, wavname in enumerate(wav_list):
        wav_path = f'C:\\Personal_Folder\\EGG_data_cut\\{name}\\{wavname}'
        # newname = i + age + sex + j
        wav_element = wavname.split('_')
        new_name = 'n' + name_element[0] + '_' + str(j) 
        new_path = os.path.join(new_folder_path, new_name)
        os.rename(wav_path, new_path)