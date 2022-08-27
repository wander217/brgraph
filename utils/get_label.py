import json

data_path = r'D:\python_project\breg_graph\data\train.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

data_path = r'D:\python_project\breg_graph\data\test.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data.extend(json.loads(f.read()))

label = {}
for file in data:
    for target in file['target']:
        label[target['label']] = 1

save_path = r'D:\python_project\breg_graph\asset\breg\label.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(list(label.keys()), indent=4))
