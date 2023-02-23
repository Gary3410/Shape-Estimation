import os
from tqdm import tqdm
txt_save_path = "/home/potato/workplace/cocoapi/PythonAPI/name_list_train.txt"
txt_save_path_val = "/home/potato/workplace/cocoapi/PythonAPI/name_list_val.txt"
fw = open(txt_save_path, "w")
fg = open(txt_save_path_val, "w")
for i in tqdm(range(420)):
    if i < 280 or i >= 350:
        for id in range(5):
            filename = str(i) +"_"+ str(id) + ".png"
            fw.write(filename + '\n')
    else:
        for id in range(5):
            filename = str(i) + "_" + str(id) + ".png"
            fg.write(filename + '\n')
