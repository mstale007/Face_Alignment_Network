import cv2
import numpy as np
import face_alignment
from skimage import io
import numpy as np
cam= cv2.VideoCapture(0)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device='cpu', flip_input=False)

while True:
    ret,frame= cam.read()

    #input = io.imread(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preds = fa.get_landmarks(frame)
    #print(preds)
    #if normal
    #ret, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    #adaptive threshoding
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    #ret, contours = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (100,0,255),1)

    #cv2.imshow("Contours drawn", np.array(preds))
    #frame[preds[0]]=(255,255,255)
    for i in preds[0]:
        frame[int(i[1]),int(i[0])]=(255,255,255)
       #print(i,i[0],i[1])
    cv2.imshow("Contours drawn", frame)
    #cv2.imshow("thresholding", thresh)
    k= cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
cam.release()