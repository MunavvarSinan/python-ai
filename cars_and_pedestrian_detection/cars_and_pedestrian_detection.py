import cv2

# sample image file
img_file = 'cars.jpeg'
# sample video file
video = cv2.VideoCapture('cars_and_pedestrian.mp4')
# pretrained data
cars_classifier = 'cars.xml'
pedestrian_classifier = 'haarcascade_fullbody.xml'
# train the data
cars_tracker = cv2.CascadeClassifier(cars_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)
# while the video is open
while True:
    # read the video and get the frame -- read_successful will return a boolean value and frame return all frames in each loopings
    read_successful, frame = video.read()
    # if it is successful only
    if read_successful:
        # convert the video to greyscale
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # exit
        break
    # get all the frames in all sizes
    cars = cars_tracker.detectMultiScale(greyscale_frame, 1.1, 1)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscale_frame, 1.1, 1)
    # loop through the values we got from cars and form a rectangle
    for(x, y, w, h) in cars:
        # create a rectangle
        cv2.rectangle(frame, (x +1, y + 2), (x + w, y+h), (255, 0, 0), 2 )
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Draw a rectangle around pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    # to create a frame
    cv2.imshow('cars_and_pedestrian_detection', frame)
    # wait the frame from closing
    key = cv2.waitKey(1)
    # close if 'q' pressed
    if key == 81 or key == 113:
        break

video.release()
