import cv2 
import matplotlib as plt
from ultralytics import YOLO
model=YOLO("yolov8n.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results= chosen_model.predict(img, classes=classes, conf=conf)
        return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results= predict(chosen_model, img, classes, conf=conf)
    
    for results in results:
        for box in results.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),(int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{results.names[int(box.cls[0])]}",(int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results
path="D:\\courses\\AI_workshop\\04_Computer_Vision\\Images"
image= cv2.imread(path+"\\car6.png")
result_img= predict_and_detect(model, image, classes=[2], conf=0.4  )
cv2.imshow("image",image)
cv2.imwrite("C:\\Users\\lawli\\OneDrive\\Desktop\\BSCE 7",result_img)
cv2.waitKey(0)