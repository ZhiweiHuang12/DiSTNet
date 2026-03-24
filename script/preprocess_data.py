import json
import numpy as np
# Define the path of the JSON file to read
data_dir = "/public/home/zhjiajun/hzw/academic_code/conditionalDM/data/"
file1 = json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/nm_nm_model/(nm_nm_model)_(2000).json"
file2 = data_dir + 'nm_nm_model_1900.json'
file3 = data_dir + 'nm_nm_model_2000.json'
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/on_off_nm/(on_off_nm)_(4000).json"
data2 = read_json(json_file)

# Define output file path
output_file = data_dir +  '(nm_nm_model)_5800.json'

# Read JSON file


# Merge JSON file contents
data1 = read_json(file1)

max_li = []
for i in range(len(data1)):
    
    max_li.append(np.array(data1[i]["counts"]).max())


data2 = read_json(file2)
data3 = read_json(file3)

# Assuming each JSON file is a list, merge lists
merged_data = data1 + data2 + data3
print("the lenght of data is",len(merged_data))
# Write merged data to new file
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)
