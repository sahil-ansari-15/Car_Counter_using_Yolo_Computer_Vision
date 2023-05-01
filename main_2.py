from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)       ## for webcam
# cap.set(3, 1280/2) # 3 = width
# cap.set(4, 720/2) # 4 = height

cap = cv2.VideoCapture("Videos/cars.mp4")       ## for video


model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "Zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")

while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)      ## Flip the image horizontally
    imgRegion = cv2.bitwise_and(img, mask)

    # results = model(img, stream=True)       ## show: add bounding boxes to the img
                                              ## stream: DOESN'T add bounding boxes to the img

    results = model(imgRegion, stream=True)   ## using "imgRegion" instead of "img" to get best results & save computation

    for r in results:
        for box in r.boxes:

            ## opencv - Bounding Box
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ## cvzone - Bounding Box
            x, y, w, h = box.xywh[0]
            x, y, w, h = int(int(x)-(w/2)), int(int(y)-(h/2)), int(w), int(h)
            bbox = x, y, w, h
            # w, h = x2-x1, y2-y1                 # bbox=(x1, x2, w, h)
            cvzone.cornerRect(img=img, bbox=bbox, l=9)

            ## class confidence
            conf = round(float(box.conf[0]), 2)        # math.ceil(box.conf[0] * 100)/100
            # cvzone.putTextRect(img, f'{conf}', (max(0, x), max(35, y)))

            ## class name
            cls = box.cls[0]
            currentClass = classNames[int(cls)]
            if currentClass in ['car','bus','truck','motorbike'] and conf>0.3:
                cvzone.putTextRect(img=img, text=f'{currentClass} {conf}', pos=(max(0, x), max(35, y)), scale=0.6, thickness=1, offset=3)

    cv2.imshow("Images-video", img)
    cv2.imshow("Images-video-imgRegion", imgRegion)
    cv2.waitKey(0)          # 1 mili sec delay

"""
Key Notes:
(1) A car (or object) will be counted only when it crosses the predefined line.
(2) We have to track the object. i.e., should know where the car has gone from frame 1 to frame 2.
    For this, we need a tracking ID. If a car is ID1 in frame 1, then it should remain ID1 in frame 2 as well (not any other ID).
    To achieve this, we need a tracker called "sort.py" (available on GitHub: https://github.com/abewley/sort).
"""