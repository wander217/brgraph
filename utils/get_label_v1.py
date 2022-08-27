import json

data_path = r'D:\python_project\breg_graph\asset\breg\label_v1.json'

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
    "COMPANY_PHONE/FAX": ["COMPANY_PHONE", "COMPANY_FAX"],
    "COMPANY_WEBSITE/EMAIL": ["COMPANY_WEBSITE/EMAIL"],
    "OWNER_TYPE": ["OWNER_TYPE"],
    "OWNER_NAME/SEX/BIRTHDAY/ETHNICITY/NATION": ["LABEL_OWNER", "OWNER_NAME", "OWNER_SEX", "OWNER_BIRTHDAY",
                                                 "OWNER_ETHNICITY", "OWNER_NATION"],
    "OWNER_IDCARD_TYPE/NUMBER": ["OWNER_IDCARD_TYPE", "OWNER_IDCARD_NUMBER"],
    "OWNER_IDCARD_DATE/PLACE": ["OWNER_IDCARD_DATE", "OWNER_IDCARD_PLACE"],
    "OWNER_RESIDENCE_PERMANENT/OWNER_LIVING_PLACE": ["OWNER_RESIDENCE_PERMANENT", "OWNER_LIVING_PLACE"],
    "BUSSINESS_CAPITAL": ["BUSSINESS_CAPITAL"]
}

del LABEL['OTHER']
with open(data_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(list(LABEL.keys()), indent=4))