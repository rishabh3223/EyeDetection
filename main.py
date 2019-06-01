import cv2
import autopy
from matplotlib import pyplot as plt
import seaborn as sns

ESCAPE_KEY = 27
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

def show_circles(hough_circles, eye_x_pos, eye_y_pos, roi_color2):
    try:
        for circle in hough_circles[0, :]:
            # Standards: DRY (don't repeat yourself), define circle_center once and use it twice.
            circle_center = (circle[0], circle[1])
            # draw the outer circle
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=circle[2],
                color=WHITE,
                thickness=2
            )
            # print("drawing circle")
            # draw the center of the circle
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=2,
                color=WHITE,
                thickness=3
            )

    except Exception as e:
        pass



face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_righteye_2splits.xml'
)

#number signifies camera
video_capture = cv2.VideoCapture(0)
eye_x_positions = list()
eye_y_positions = list()



while 1:
    success, image = video_capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)

    for (eye_x, eye_y, eye_width, eye_height) in eyes:
        cv2.rectangle(
            img=image, 
            pt1=(eye_x, eye_y), 
            pt2=(eye_x + eye_width, eye_y + eye_height), 
            color=GREEN, 
            thickness=2
        )
        roi_gray2 = gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
        roi_color2 = image[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
        hough_circles = cv2.HoughCircles(
            roi_gray2,
            cv2.HOUGH_GRADIENT,
            1,
            200,
            param1=200,
            param2=1,
            minRadius=0,
            maxRadius=0
        )

        eye_x_pos = (eye_x + eye_width) / 2
        eye_y_pos = (eye_y + eye_height) / 2
        print(eye_x_pos, eye_y_pos)
        eye_x_positions.append(eye_x_pos)
        eye_y_positions.append(eye_y_pos)
        
        show_circles(hough_circles, eye_x_pos, eye_y_pos, roi_color2)

    cv2.imshow('img', image)
    key_pressed = cv2.waitKey(30) & 0xff
    if key_pressed == ESCAPE_KEY:
        break


video_capture.release()
cv2.destroyAllWindows()
data = list(zip(eye_x_positions, eye_y_positions))

print(data)

plt.scatter(eye_x_positions, eye_y_positions)
plt.title("Eye Positions")
plt.show()