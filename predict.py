
import numpy as np
import cv2
from collections import deque
from numpy import random
import GUI

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
object_counter = {}

mask_ROI = np.zeros_like(frame)
cv2.fillPoly(mask_ROI, [pts], (128, 255, 0))
cv2.polylines(mask_ROI, [pts], True, (255, 255, 128), 2)
cv2.line(frame, line2[0], line2[1], (255, 0, 112), 3)
frame = cv2.addWeighted(frame, 1, mask_ROI, 0.2, 0)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = imutils.resize(frame, width=1000)
image = Image.fromarray(frame)
photo = ImageTk.PhotoImage(image=image)
label.config(image=photo)
label.image = photo

root.update()