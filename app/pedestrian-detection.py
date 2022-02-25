import cv2

# Imports for image procesing
from PIL import Image

# Imports for prediction
from predict import initialize, predict_image


def main():
    # Load and intialize the model
    initialize()
    # create a video capture object
    capture = cv2.VideoCapture(0)
    while(True):
        # capture the video frame by frame
        ret, frame = capture.read()
        # pass the frame for detection
        predictions = predict_image(Image.fromarray(frame))
        # display the frame
        cv2.imshow('Pedestrian detector', frame)
        # define quitting button
        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:
            break
    # release the object once the loop is over
    capture.release()
    # destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()