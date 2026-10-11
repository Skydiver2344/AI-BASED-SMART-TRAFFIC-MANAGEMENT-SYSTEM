import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model for object detection
model=YOLO("C:/Users/divya/OneDrive/Desktop/MINOR PROJECT 5th sem/Code/yolov8n.pt")

# Class IDs for car=2, motorcycle=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]  

#Opening the video file
video=cv2.VideoCapture("C:/Users/divya/OneDrive/Desktop/MINOR PROJECT 5th sem/Code/video.mp4")

#continuously reading the video frame by frame and processing each frame for vehicle detection
while True:
     
     #reads the video frame by frame and if it is not successful, it breaks the loop
    success,frame=video.read()
    if not success:
        break                      

    # Run YOLO object detection on the current frame    
    results=model(frame)

    #Detects class IDs of detected objects
    classes = results[0].boxes.cls

    #generates boxes around detected vehicles
    annotated_frame=frame.copy()       
    
    #extracts the bounding box coordinates of detected vehicles and calculates their center points, then draws a red dot at the center of each detected vehicle
    boxes = results[0].boxes.xyxy
    for box,cls in zip(boxes, classes):
        if int(cls) not in vehicle_classes:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        #creates a red dot at centre of annotated frame
        cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  

         # Get the detected vehicle's name
        label = model.names[int(cls)]
        cv2.putText(annotated_frame, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (0, 0, 255), 2 )     

    #counts the number of vehicles detected in each frame and prints the class IDs and total count of detected objects
    vehicle_count=0
    for cls in classes:
        if int(cls) in vehicle_classes:
            vehicle_count += 1
    cv2.putText(annotated_frame,f"Vehicles: {vehicle_count}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)          

    #Displays the annotated frame with detected vehicles and their center points in a window named "Video"     
    cv2.imshow("Video",annotated_frame)     
    
    #Makes video close when 'q' is pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break                              

#closees video window
video.release()
cv2.destroyAllWindows()                     

