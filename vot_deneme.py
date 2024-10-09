import cv2
import numpy as np
import vot
import sys,os
import time

handle = vot.VOT("rectangle")
selection = handle.region()

colorimage = handle.frame()
if not colorimage:
    sys.exit(0)

frame = cv2.cvtColor(cv2.imread(colorimage),cv2.COLOR_BGR2RGB)
tracker = cv2.legacy.TrackerMOSSE_create()
bbox = (selection.x, selection.y, selection.width, selection.height)
target_pos = np.array([bbox[0] + int(bbox[2]/2), bbox[1] + int(bbox[3]/2)])
target_sz = np.array([bbox[2], bbox[3]])

status_tracker = tracker.init(frame, bbox)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    frame_cp = cv2.cvtColor(cv2.imread(imagefile),cv2.COLOR_BGR2RGB)
    status_tracker, bbox = tracker.update(frame_cp)

    handle.report(vot.Rectangle(bbox[0],bbox[1],bbox[2],bbox[3]))