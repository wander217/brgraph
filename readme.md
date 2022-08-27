# Cài Đặt
Để cài đặt thực hiện gõ lệnh sau
```
pip install -r requirements.txt
```
Link pretrained model:
```
"2 layer": https://drive.google.com/file/d/1tG4-yznwO4AXrkxoWh5QttGrRsScpa1D/view?usp=sharing 
```
# Class BREGPredictor trong predictor.py
Dữ liệu khởi tạo bao gồm:
```
{
    "config": Đường dẫn tới file config (trong asset/breg),
    "alphabet": Đường dẫn tới file alphabet (trong asset/breg),
    "label": Đường dẫn tới file label (trong asset/breg),
    "pretrained": Đường dẫn tới file pretrained (checkpoint50),
}
```
Gọi hàm predict với đầu vào là một list các bounding box.
Với mỗi phần tử có dạng:
```
{
    "file_name":  Tên file ảnh,
    "shape": Kích thước file ảnh,
    "target": 
    {
        "bbox": Tọa độ của bounding box gồm 4 đỉnh dự đoán từ detection,
        "bbox_score": Độ tin cậy của bounding box,
        "text": phần text dự đoán từ mạng recognize,
        "text_score": Độ tin cậy của text,
    }
}
```
Dữ liệu trả về:
```
{
    {
    "file_name":  Tên file ảnh,
    "shape": Kích thước file ảnh,
    "target": 
    {
        "bbox": Tọa độ của bounding box gồm 4 đỉnh dự đoán từ detection,
        "bbox_score": Độ tin cậy của bounding box,
        "text": phần text dự đoán từ mạng recognize,
        "text_score": Độ tin cậy của text,
        "label": Nhãn của bounding box,
        "label_score": Độ tin cậy của nhãn
    }
}
```
# Predict bằng câu lệnh:
Cấu trúc câu lệnh:
```
{
    py predictor.py 
    -c [Đường dẫn tới file config yaml: lưu trong thư mục asset/breg]
    -r [Đường dẫn tới file checkpoint: file checkpoint50.pth]
    -i [Đường dẫn tới file json chứa data: lưu trong thư mục data]
    -a [Đường dẫn tới file alphabet: lưu trong thư mục asset/breg]
    -l [Đường dẫn tới file label: lưu trong thư mục asset/breg]
}
```
Kết quả in ra:
```
{
    -----------------------------------------------------
    Pred: [Dòng chữ được dự đoán] [Nhãn dự đoán] [Độ tin cậy của nhãn]
    GT: [Dòng chữ đích] [Nhãn đích]
    -----------------------------------------------------
}