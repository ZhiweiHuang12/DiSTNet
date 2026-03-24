import json
import numpy as np
# Define the path of the JSON file to read
# Read JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def get_new_data(data):
    new_data = []
    for i in range(len(data)):
        temp_dict = {'parameter':data[i]["parameter"],"counts":np.array(data[i]["counts"])[:,:2,:].tolist()}
        new_data.append(temp_dict)
    return new_data

data_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/nm_nm_model/"
file1 = json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/nm_nm_model/(nm_nm_model)_(2000).json"
file2 = data_dir + 'nm_nm_model_1900.json'
file3 = data_dir + 'nm_nm_model_1950.json'
# Define output file path
output_file = data_dir +  '(nm_nm_model)_(5870).json'
# Merge JSON file contents
data1 = read_json(file1)
new_data1 = get_new_data(data1)

# new_data1 = get_new_data(data1)
# with open(output_file, 'w') as f:
#     json.dump(new_data1, f)

data2 = read_json(file2)
data3 = read_json(file3)


new_data2 = get_new_data(data2)
new_data3 = get_new_data(data3)


# merged_data=[]
# merged_data.expand(new_data1)

# Assuming each JSON file is a list, merge lists
merged_data = new_data1 + new_data2 + new_data3


print("the lenght of data is",len(merged_data))
# Write merged data to new file
with open(output_file, 'w') as f:
    json.dump(merged_data, f)


# new_data = []
# for i in range(len(data1)):
#     temp_dict = {'parameter':data1[i]["parameter"],"counts":np.array(data1[0]["counts"])[:,:2,:]}
#     new_data.append(temp_dict)


# max_li = []
# for i in range(len(data1)):
    
#     max_li.append(np.array(data1[i]["counts"]).max())