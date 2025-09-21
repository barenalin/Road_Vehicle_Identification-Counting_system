# Traffic Analysis with YOLOv5

這是一個基於 **YOLOv5** 的交通流量分析小專題，能偵測影片中的車輛，並分別統計南北向的車流量。  
此專案同時結合 **SORT 演算法** 進行目標追蹤，確保車輛計數的準確性。  

## ✨ 功能特色
- 車輛偵測（YOLOv5）
- 車輛追蹤（SORT）
- 南北向車流量計數
- 即時顯示偵測結果

## ⚠️ 限制
- 使用者需自行準備影片檔案（例如 `highway.mp4`）
- 中線需手動設定作為計數依據
- YOLO 權重檔（如 `yolov5s.pt`）需自行下載

## 📦 環境需求

請先安裝所需套件：
```bash
pip install -r requirements.txt

