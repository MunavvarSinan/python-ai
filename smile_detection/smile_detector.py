import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
# eye_detector = cv2.CascadeClassifier('haarcscade_eye.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(grayscale)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y+h), (100, 200, 50), 4)
        # get the subframe using numpy N-dimentional array slicing
        the_face = frame[y:y+h, x:x+w]
        # the_face = (x, y, w, h)
        face_greyscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(
            face_greyscale, scaleFactor=1.7, minNeighbors=20)
        # eyes = eye_detector.detectMultiScale(
        #     face_greyscale, scaleFactor=1.7, minNeighbors=20)
        # find all the smiles in the faces
        # for(x_, y_, w_, h_) in eyes:
        #     cv2.rectangle(frame, (x_, y_), (x_ + w_, y_+h_), (50, 50, 200), 4)
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    cv2.imshow('smile_detector', frame)
    key = cv2.waitKey(1)
    # if the 'q' key is pressed, stop the loop
    if key == 81 or key == 113:
        break

# release the webcam
webcam.release()
cv2.destroyAllWindows()
