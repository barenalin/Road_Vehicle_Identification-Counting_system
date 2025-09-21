import cv2
import torch
import numpy as np
import pandas as pd
import sys
from threading import Thread
from queue import Queue
from pathlib import Path

sys.path.append('./sort')
from sort import Sort

# 載入你訓練好的 YOLO 模型
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weight" / "best.pt"            

OUTPUT_DIR   = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)                        
OUTPUT_PATH  = OUTPUT_DIR / "output.avi"

# 讓 Python 找得到 sort 模組
sys.path.append(str(ROOT / "sort"))
from sort import Sort

# -----------------------------
# 載入訓練好的 YOLO 模型（torch.hub）
# -----------------------------
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"weights not found: {WEIGHTS_PATH}")

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=str(WEIGHTS_PATH),
    force_reload=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 設置 YOLO 模型參數
model.conf = 0.1  # confidence 閾值
model.iou = 0.7   # IOU 閾值

tracker = Sort(max_age=3600, min_hits=1, iou_threshold=0.75)  # (未匹配的最大幀數, 最小命中次數)

# 非極大值抑制（NMS）函數 
def non_max_suppression(detections, iou_threshold):
    if len(detections) == 0:
        return detections

    boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values
    scores = detections['confidence'].values
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=model.conf, nms_threshold=iou_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        return detections.iloc[indices]
    else:
        return detections.iloc[[]]

def is_point_above_line(point, line):
    x, y = point
    x1, y1, x2, y2 = line
    position = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return position < 0  

def process_frame(input_queue, output_queue):
    while True:
        frame = input_queue.get()
        if frame is None:  # 檢查是否已完成所有處理
            break
        frame = cv2.resize(frame, (640, 1080))
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 添加高斯平滑
        # 偵測車輛
        results = model(frame, size=640)
        detections = results.pandas().xyxy[0]
        cars = detections[detections['name'].isin(['car', 'truck', 'bus'])]  # 篩選車輛類別

        # 過濾重疊框
        cars = non_max_suppression(cars, iou_threshold=0.75)

        # 準備 SORT 追蹤輸入
        sort_input = []
        for _, row in cars.iterrows():
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            confidence = row['confidence']
            sort_input.append([x1, y1, x2, y2, confidence])
        sort_input = np.array(sort_input)

        # 確保輸入不為空
        if len(sort_input) == 0:
            tracks = np.empty((0, 5))  # 若無檢測結果，返回空陣列
        else:
            tracks = tracker.update(sort_input)

        # 將結果放入輸出佇列
        output_queue.put((frame, tracks))

# 顯示影片的執行緒函數
def display_frame(output_queue, line_coordinates, out):
    paused = False
    count = 0
    L_list = [0] * 5
    R_list = [0] * 5
    MaxL = 0
    MaxR = 0
    while True:
        try:
            if not paused:
                frame, tracks = output_queue.get(timeout=5)

            # 調整視窗大小
            frame = cv2.resize(frame, (640, 1080))

            # 畫中線
            x1, y1, x2, y2 = line_coordinates
            overlay = frame.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # 計算車輛在 Left 和 Right
            left_count = 0
            right_count = 0
            
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if is_point_above_line((center_x, center_y), line_coordinates):
                    left_count += 1
                else:
                    right_count += 1

                # 在透明圖層繪製框線與文字
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)

            L_list[count] = left_count
            R_list[count] = right_count
            count += 1
            
            if(count == 4):
                count = 0
                MaxL = max(L_list)
                MaxR = max(R_list)
                
            
            # 合併透明圖層到原始影像
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 顯示計數

            cv2.putText(frame, f"Left: {MaxL}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right: {MaxR}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 寫入輸出影片
            out.write(frame)

            # 顯示影片
            cv2.imshow('Vehicle Detection and Counting', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused

        except:
            break

# 讀取影片
video_path   = ROOT / "videos" / "highway_slowmotion.mp4"  
cap = cv2.VideoCapture(video_path)

# 確保影片成功打開
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# 獲取影片的幀率，用於設置輸出視頻幀率
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_coordinates = (0, 720, 600, 0)  # 定義分隔線的起點和終點

# 定義影片輸出
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 1080))

# 建立佇列
input_queue = Queue(maxsize=10)
output_queue = Queue(maxsize=10)

# 啟動執行緒
process_thread = Thread(target=process_frame, args=(input_queue, output_queue))
display_thread = Thread(target=display_frame, args=(output_queue, line_coordinates, out))

process_thread.start()
display_thread.start()

# 將幀送入佇列
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_queue.put(frame)

# 停止執行緒
input_queue.put(None)
process_thread.join()
display_thread.join()

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
