import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not
from datetime import datetime
import csv
 
mask_path = r"C:\Users\renad\Desktop\My-Valet-main\My-Valet-main\static\mask_1920_1080.png"
video_path = r"C:\Users\renad\Downloads\parking_1920_1080.mp4"
 
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)
 
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
 
spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None
frame_number = 0
step = 30
 
COST_PER_HOUR = 10   
parking_log = [{} for _ in spots]   
 
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break
 
    if frame_number % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x, y, w, h = spot
            spot_crop = frame[y:y + h, x:x + w, :]
            prev_crop = previous_frame[y:y + h, x:x + w, :]
            diffs[spot_index] = np.abs(np.mean(spot_crop) - np.mean(prev_crop))
 
    if frame_number % step == 0:
        if previous_frame is None:
            check_indices = range(len(spots))
        else:
            check_indices = [i for i in np.argsort(diffs) if diffs[i] / np.max(diffs) > 0.4]

        for spot_index in check_indices:
            x, y, w, h = spots[spot_index]
            spot_crop = frame[y:y + h, x:x + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status
 
    if frame_number % step == 0:
        previous_frame = frame.copy()
 
    for spot_index, spot_status in enumerate(spots_status):
        log = parking_log[spot_index]
        if spot_status and not log.get("occupied", False):
            log["occupied"] = True
            log["start_time"] = datetime.now()

        elif not spot_status and log.get("occupied", False):
            log["occupied"] = False
            log["end_time"] = datetime.now()
            duration_seconds = (log["end_time"] - log["start_time"]).total_seconds()
            hours = duration_seconds / 3600
            log["duration"] = round(hours, 2)
            log["cost"] = round(hours * COST_PER_HOUR, 2)
            print(f"Spot {spot_index}: Parked for {log['duration']} hours, Cost = {log['cost']} EGP")
 
    for spot_index, spot in enumerate(spots):
        x, y, w, h = spot
        status = spots_status[spot_index]
        color = (0, 255, 0) if status else (0, 0, 255)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
 
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame,
                f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
 
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_number += 1
 
cap.release()
cv2.destroyAllWindows()
 
with open("parking_log.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Spot ID", "Start Time", "End Time", "Duration (h)", "Cost (EGP)"])
    for i, log in enumerate(parking_log):
        if "end_time" in log:
            writer.writerow([
                i,
                log["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                log["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
                log["duration"],
                log["cost"]
        ])
