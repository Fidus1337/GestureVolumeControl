import cv2
import time
import math
from HandDetector_module import HandsDetector

# Modules for changing the volume of computer
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


def initialize_volume():
    """Initialize system audio endpoint volume interface."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def draw_fps(img, ptime):
    """Draw current FPS on the image."""
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    cv2.putText(img, f"FPS: {int(fps)}", (25, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    return ctime


def draw_calibration_text(img, blink):
    """Draw blinking instruction text during calibration."""
    if blink:
        cv2.putText(img, "Show max length between thumb and index",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "and press SPACE", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def draw_hand_graphics(img, x4, y4, x8, y8, norm_distance):
    """Draw visual hand markers and volume feedback on the image."""
    # Draw line and circles between fingers
    cv2.circle(img, (x4, y4), 6, (0, 255, 0), cv2.FILLED)
    cv2.circle(img, (x8, y8), 6, (0, 255, 0), cv2.FILLED)
    cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 2)

    # Draw average circle in red or green based on normalized distance
    color = (0, 0, 255) if norm_distance < 20 else (0, 255, 0)
    cx, cy = (x4 + x8) // 2, (y4 + y8) // 2
    cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)

    # Display normalized distance percentage
    cv2.putText(img, f"Norm Dist: {int(norm_distance)}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Set up camera and hand detector
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

ptime = 0
detector = HandsDetector(max_num_hands=1)
volume = initialize_volume()

max_distance_calibrated = None
calibrated = False
blink = True
last_blink_time = 0
blink_interval = 0.7

while True:
    success, img = cap.read()
    if not success:
        break

    detector.detect_hands(img)
    lm_list = detector.findPositions(img)

    if not calibrated:
        # Blinking text with instruction
        current_time = time.time()
        if current_time - last_blink_time > blink_interval:
            blink = not blink
            last_blink_time = current_time

        draw_calibration_text(img, blink)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # space button to set maximum value between thumb and index fingers
            if lm_list and len(lm_list) > 8:
                x4, y4 = lm_list[4][1], lm_list[4][2]
                x8, y8 = lm_list[8][1], lm_list[8][2]
                max_distance_calibrated = calculate_distance((x4, y4), (x8, y8))
                calibrated = True
                print(f"Calibrated max distance: {max_distance_calibrated:.2f}")

        # Press 'q' to quit the program
        if key == ord('q') or key == ord('Q'):
            print("YOU PRESSED 'Q' TO STOP PROGRAM")
            break

        cv2.imshow("Result", img)
        continue

    # After choosing right maximum distance between thumb and index, the user starts to change sound volume
    if lm_list and len(lm_list) > 8:
        x4, y4 = lm_list[4][1], lm_list[4][2]
        x8, y8 = lm_list[8][1], lm_list[8][2]

        dist_4_8 = calculate_distance((x4, y4), (x8, y8))

        # Normalize distance between fingers
        norm_distance = (dist_4_8 / max_distance_calibrated) * 100
        norm_distance -= 10  # adjustment
        norm_distance = max(0, min(100, norm_distance))

        # Set system volume based on normalized distance
        volume.SetMasterVolumeLevelScalar(norm_distance / 100.0, None)

        # Draw visual indicators
        draw_hand_graphics(img, x4, y4, x8, y8, norm_distance)
    else:
        calibrated = False

    # Draw FPS
    ptime = draw_fps(img, ptime)

    # To quit program at all
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("YOU PRESSED 'Q' TO STOP PROGRAM")
        break
    
    # To stop changing the value
    if key == ord('s'):
        calibrated = False
        print("YOU PRESSED 'S' TO STOP CHANGING VOLUME")
        continue
        
    cv2.imshow("Result", img)

cap.release()
cv2.destroyAllWindows()
