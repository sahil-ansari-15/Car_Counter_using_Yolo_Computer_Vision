from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0)       ## input from webcam
# cap.set(3, 1280/2) # 3 = width
# cap.set(4, 720/2) # 4 = height

cap = cv2.VideoCapture("Videos/cars.mp4")       ## input for video

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

mask = cv2.imread("Images/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting
limits = [400, 297, 673, 297]             # coordinates of counting line (left & right end points)
totalCount = []

while True:                                   # looping over images
    success, img = cap.read()
    # img = cv2.flip(img, 1)      ## Flip the image horizontally
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("Images/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(imgBack=img, imgFront=imgGraphics, pos=[0, 0])

    # results = model(img, stream=True)       ## show: add bounding boxes to the img
                                              ## stream: DOESN'T add bounding boxes to the img
    results = model(imgRegion, stream=True)   ## using "imgRegion" instead of "img" to get best results & save computation

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:                     # looping over detected objects in a image

            ## opencv - Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ## cvzone - Bounding Box
            # x, y, w, h = box.xywh[0]
            # x1, y1, w, h = int(int(x)-(w/2)), int(int(y)-(h/2)), int(w), int(h)
            # x2, y2 = x1+w, y1+h
            bbox = x1, y1, w, h
            # cvzone.cornerRect(img=img, bbox=bbox, l=9, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0))

            ## class confidence
            conf = round(float(box.conf[0]), 2)        # math.ceil(box.conf[0] * 100)/100
            # cvzone.putTextRect(img, f'{conf}', (max(0, x), max(35, y)))

            ## class name
            cls = box.cls[0]
            currentClass = classNames[int(cls)]

            if currentClass in ['car','bus','truck','motorbike'] and conf>0.3:
                # cvzone.putTextRect(img=img, text=f'{currentClass} {conf}', pos=(max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img=img, bbox=bbox, l=9, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    ## Tracking
    resultsTracker = tracker.update(detections)

    cv2.line(img=img, pt1=(limits[0], limits[1]), pt2=(limits[2], limits[3]),
             color=(0, 0, 255), thickness=2)  # displaying red counting line


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        ## displaying object tracking ID & bounding box
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img=img, text=f'{int(id)}', pos=(max(0, x1), max(35, y1)),
                           scale=1, thickness=1, offset=3,
                           colorT=(0, 0, 0),
                           colorR=(255, 0, 255)
                           )

        ## Counting
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)    # changing counting line color from red to green when even new object is counted.

    ## display total car count
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Images-video", img)
    # cv2.imshow("Images-video-imgRegion", imgRegion)
    cv2.waitKey(0)          # 1 mili sec delay


## Yolo sample code for object detection in a image
# from ultralytics import YOLO
# import cv2
# model = YOLO('../Yolo-Weights/yolov8m.pt')
# results = model("Images/1.png", show=True)
# cv2.waitKey(0)