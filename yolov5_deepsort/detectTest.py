import tracker
from detector import Detector
import cv2



# init yolov5
detector = Detector()
im=cv2.imread('./video/image2.jpg')


# set size，1920x1080->960x540
im = cv2.resize(im, (960, 540))
bboxes = detector.detect(im)

for (x1, y1, x2, y2, label, conf) in bboxes:
    color = (0, 255, 0)  # green
    label_with_conf = f'{label}: {conf:.2f}'

    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)



#show the image with box
cv2.imshow('Detected Objects', im)
cv2.waitKey(0)
cv2.destroyAllWindows()





