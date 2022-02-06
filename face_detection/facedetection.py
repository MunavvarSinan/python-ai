import cv2

# load some pretrained data from opencv(haarcascade)
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Test with an image
# img = cv2.imread('someimage.jpg')

# To capture the video from webcam
webcam = cv2.VideoCapture(0)

# while our webcam is open
while True:
    # read our webcam --- successful_frame_read which return a boolean value and frame is the image
    successful_frame_read, frame = webcam.read()
    # change the color to gray
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    """detect the faces -- detectMultiScale is a function of the trained_data
       that used to detect all the faces with different sizes.
       it return a value like this [[140 182 577 577]] which the first value is the x-axis 
       which represents the top left corner of the face and the last value is the bottom right corner of the face 
       140 -- x axis of the top left corner of the face -- which actually represents the  top left corner of the face
       182 -- y axis of the top left corner of the face -- which actually represents the bottom right corner of the face
       577 -- x axis of the bottom right corner of the face -- the width of the face
       577 -- y axis of the bottom right corner of the face -- the height of the face    """
       
    detect_faces = trained_data.detectMultiScale(grayscale_img)
    # draw a rectangle around the faces by looping through the detect_faces
    for(x, y, w, h) in detect_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
    # use cv2.imshow to show the image in a window
    cv2.imshow('face detector', frame)
    # use cv2.waitKey to hold the window open until a key is pressed (1) is window is refreshed in each 1 millisecond or else we manually press a button to move from one frame to another
    key = cv2.waitKey(1)
    # if the 'q' key is pressed, stop the loop
    if key == 81 or key == 113:
        break

# release the webcam
webcam.release()
