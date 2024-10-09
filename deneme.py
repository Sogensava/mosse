import cv2
import numpy as np

misato_path = "/home/atila/Pictures/misato/"

display_name = 'Sequence'
sequence_length = 300

frame_number = "00000000"
input_string = misato_path+frame_number+'.bmp'
frame = cv2.imread(input_string)

bbox = cv2.selectROI(frame, False, False)

cv2.destroyAllWindows()

#tracker = cv2.legacy.TrackerMedianFlow_create()
tracker = cv2.legacy.TrackerMOSSE_create()
status_tracker = tracker.init(frame, bbox)

while True:

    input_string = misato_path+frame_number+'.bmp'
    print(input_string)
    frame = cv2.imread(input_string)

    if int(frame_number) == sequence_length:
        break
    
    frame_disp = frame.copy()

    status_tracker, bbox = tracker.update(frame)

    x, y, w, h = [int(i) for i in bbox]
    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 5)

    font_color = (0, 0, 0)
    cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)
    cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)
    cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                font_color, 1)

    frame_number += f"{int(frame_number) + 1:08d}"
    frame_number = frame_number[-8:]
    
    cv2.imshow(display_name,frame_disp)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()