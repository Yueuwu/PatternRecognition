import cv2

face_cascade = cv2.CascadeClassifier(".\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(".\haarcascades\haarcascade_eye.xml")
#change to gray
img = cv2.imread(".\somefaces\oneface1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect objects(head). for better spotting increase or decrease second and third argument
faces = face_cascade.detectMultiScale(gray, 2, 5)
#highlight objects
for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    # looking for eyes
    roi_color = img[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]
    # detect objects(eyes). for better spotting increase or decrease second and third argument
    eyes = eye_cascade.detectMultiScale(roi_gray, 5, 4)
    for ex, ey, ew, eh in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# display window with img
cv2.imshow('Face detected', img)
#close window after click
cv2.waitKey(0)
cv2.destroyAllWindows()
