import json
import os

data_root = r'C:\Users\thinhtq\Downloads\test_data_json_graph\test_data_json_graph'
data = []
for file in os.listdir(data_root):
    with open(os.path.join(data_root, file), 'r', encoding='utf-8') as f:
        item_data = json.loads(f.read())
        tmp = item_data['target']
        if len(tmp) == 0:
            continue
        for i in range(len(tmp)):
            if ("bussiness" in tmp[i]['label'] and "bussiness_capital" != tmp[i]['label']) \
                    or ("business" in tmp[i]['label']) or ("shareholder" in tmp[i]['label']):
                print(tmp[i]['label'])
                tmp[i]['label'] = 'other'
            elif "company_english_name" == tmp[i]['label'] or "company_short_name" == tmp[i]['label']:
                tmp[i]['label'] = 'company_vietnamese_name'
        data.append({
            "file_name": ".".join(file.split(".")[:-1]),
            "target": tmp
        })

save_path = r'D:\python_project\breg_graph\data\test.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=True))
