import os

path =r'C:\Personal_Folder\EGG_data_renamed'

name = os.listdir(path)

for i in name:
    path2 = f'C:\\Personal_Folder\\EGG_data_renamed\\{i}'
    new_path = path2 + '.wav'
    os.rename(path2, new_path)