import numpy as np
import cv2
import tracker
from detector import Detector

if __name__ == '__main__':
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    detector = Detector()
    capture = cv2.VideoCapture('./video/school1.mp4')

    # Determine the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
    out = cv2.VideoWriter('./output/output.mp4', fourcc, 20.0, (960, 540))  # Output file, codec, fps, and frame size

    while True:
        _, im = capture.read()
        if im is None:
            break

        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
        else:
            output_image_frame = im

        cv2.imshow('deepsort', output_image_frame)
        out.write(output_image_frame)  # Write the processed frame to the output file
        cv2.waitKey(1)

    capture.release()
    out.release()  # Close the VideoWriter object
    cv2.destroyAllWindows()
