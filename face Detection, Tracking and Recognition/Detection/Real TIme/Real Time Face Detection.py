import cv2
import numpy as np


net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb','opencv_face_detector.pbtxt')

capture = cv2.VideoCapture(0)

while True:

    ret ,frame = capture.read()

    (h , w ) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),[104.,117.,123],False,False)
    net.setInput(blob)
    detections = net.forward()

    detected_face = 0

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence >0.7:
            detected_face +=1
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX , startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY -10 >10 else startY + 10
            cv2.rectangle(frame,(startX,startY),(endX,endY),(200,20,30),2)
            text = "{:.3f}%".format(confidence*100)
            cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow("Face Detection",frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        print(detections[0,0,0])
        break

capture.release()
cv2.destroyAllWindows()