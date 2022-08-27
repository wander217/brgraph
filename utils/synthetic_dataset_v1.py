import json
import os

LABEL = {
    "OTHER": ["COMMITTEE", "OTHER",
              "BUSINESS_TYPE", "LABEL_BUSINESS_TYPE",
              "BUSSINESS_TYPE_ORDER", "BUSSINESS_TYPE_NAME",
              "BUSSINESS_TYPE_CODE", "LABEL_SHAREHOLDER",
              "SHAREHOLDER_ORDER", "SHAREHOLDER_NAME",
              "SHAREHOLDER_ADDRESS", "SHAREHOLDER_CAPITAL",
              "SHAREHOLDER_RATE", "SHAREHOLDER_IDCARD",
              "SHAREHOLDER_NOTE", "SHAREHOLDER_NATIONAL"],
    "CONTRACT_TYPE": ["CONTRACT_TYPE"],
    "REGISTER_DATE": ["REGISTER_DATE"],
    "COMPANY_CODE": ["COMPANY_CODE"],
    "COMPANY_NAME": ["COMPANY_VIETNAMESE_NAME", "LABEL_COMPANY_NAME", "COMPANY_ENGLISH_NAME", "COMPANY_SHORT_NAME"],
    "COMPANY_ADDRESS": ["COMPANY_ADDRESS"],
    "COMPANY_PHONE/FAX/WEBSITE/EMAIL": ["COMPANY_PHONE", "COMPANY_FAX", "COMPANY_WEBSITE/EMAIL"],
    "OWNER_TYPE": ["OWNER_TYPE"],
    "OWNER_NAME/SEX/BIRTHDAY/ETHNICITY/NATION": ["LABEL_OWNER", "OWNER_NAME", "OWNER_SEX", "OWNER_BIRTHDAY",
                                                 "OWNER_ETHNICITY", "OWNER_NATION"],
    "OWNER_IDCARD_TYPE/NUMBER/DATE/PLACE": ["OWNER_IDCARD_TYPE", "OWNER_IDCARD_NUMBER",
                                            "OWNER_IDCARD_DATE", "OWNER_IDCARD_PLACE"],
    "OWNER_RESIDENCE_PERMANENT/LIVING_PLACE": ["OWNER_RESIDENCE_PERMANENT", "OWNER_LIVING_PLACE"],
    "BUSSINESS_CAPITAL": ["BUSSINESS_CAPITAL"]
}

# data_root = r'F:\project\python\brgraph\data_v1\train_data_json_graph'
data_root = r'F:\project\python\brgraph\data_v1\test_data_json_graph'
data = []


def get_label(current_label):
    for label1 in current_label:
        for label2, label3 in LABEL.items():
            if label1.upper() in label3:
                return label2
    print("-"*100)
    print(current_label)
    print("-" * 100)
    return ''


for file in os.listdir(data_root):
    with open(os.path.join(data_root, file), 'r', encoding='utf-8') as f:
        item_data = json.loads(f.read())
        tmp = item_data['target']
        if len(tmp) == 0:
            continue
        for i in range(len(tmp)):
            tmp_label = get_label(tmp[i]['label'])
            tmp[i]['label'] = tmp_label
        data.append({
            "file_name": ".".join(file.split(".")[:-1]),
            "target": tmp
        })

# save_path = r'F:\project\python\brgraph\data_v1\train.json'
save_path = r'F:\project\python\brgraph\data_v1\test.json'
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=True))
