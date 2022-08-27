import json
import os.path
import random

data_path = r'D:\python_project\breg_graph\data\train.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())
    print(len(data))

data_path = r'D:\python_project\breg_graph\data\test.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())
    print(len(data))

# save_root = r'D:\workspace\project\breg_graph\data1\splitted'
# random.shuffle(data_v1)
# total_len = len(data_v1)
# train_len = int(total_len * 0.8)
# valid_len = int(total_len * 0.1)
# train_data = data_v1[:train_len]
# valid_data = data_v1[train_len:train_len + valid_len]
# test_data = data_v1[train_len + valid_len:]
#
#
# with open(os.path.join(save_root, 'train.json'), 'w', encoding='utf-8') as f:
#     print(len(train_data))
#     f.write(json.dumps(train_data, indent=4))
#
# with open(os.path.join(save_root, 'valid.json'), 'w', encoding='utf-8') as f:
#     print(len(valid_data))
#     f.write(json.dumps(valid_data, indent=4))
#
# with open(os.path.join(save_root, 'test.json'), 'w', encoding='utf-8') as f:
#     print(len(test_data))
#     f.write(json.dumps(test_data, indent=4))
