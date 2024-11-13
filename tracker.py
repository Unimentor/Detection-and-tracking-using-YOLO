import cv2 
import matplotlib as plt
from ultralytics import YOLO
model=YOLO("YOLO MODEL") #insert yolo model

def predict(chosen_model, img, classes=[], conf=0.5): #confidence level set to 0.5
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
path="Enter path" #add one more backward slash with each one in path, e.g. c:\\users\\desktop\\...
image= cv2.imread(path+"\\file name")
result_img= predict_and_detect(model, image, classes=[], conf=0.4  ) #enter arguments in classes for detection of objects
#detailed classes here https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
cv2.imshow("image",image)
cv2.imwrite("enter path",result_img)
cv2.waitKey(0)
