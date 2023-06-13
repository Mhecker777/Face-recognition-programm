import cv2

# XML parsing
cascPath = 'default.xml'

# Cascade for face
faceCascade = cv2.CascadeClassifier(cascPath)

# Opening first available camera
videoCapture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = videoCapture.read()
    # Detect many faces in 1 frame
    faces = faceCascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('VisageDetector', frame)

    # If 'q' symbol pressed - break cycle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
videoCapture.release()
cv2.destroyAllWindows()
