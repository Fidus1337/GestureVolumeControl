import cv2
import time
import math
from HandDetector_module import HandsDetector

# Modules for changing the volume of computer
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Setting up settings for sound
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

ptime = 0
detector = HandsDetector(max_num_hands=1)

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
        # Мигающий текст с инструкцией
        current_time = time.time()
        if current_time - last_blink_time > blink_interval:
            blink = not blink
            last_blink_time = current_time

        if blink:
            cv2.putText(img, "Show max length between thumb and index",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "and press SPACE", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # пробел
            if lm_list and len(lm_list) > 8:
                x4, y4 = lm_list[4][1], lm_list[4][2]
                x8, y8 = lm_list[8][1], lm_list[8][2]
                max_distance_calibrated = math.hypot(x8 - x4, y8 - y4)
                calibrated = True
                print(f"Calibrated max distance: {max_distance_calibrated:.2f}")

        cv2.imshow("Result", img)
        continue

    # После калибровки — основной режим
    if lm_list and len(lm_list) > 8:
        x4, y4 = lm_list[4][1], lm_list[4][2]
        x8, y8 = lm_list[8][1], lm_list[8][2]

        # Рисуем точки и линию между пальцами
        cv2.circle(img, (x4, y4), 6, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x8, y8), 6, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 2)

        dist_4_8 = math.hypot(x8 - x4, y8 - y4)

        # Нормализуем расстояние
        norm_distance = (dist_4_8/max_distance_calibrated) * 100
        norm_distance-=10
        norm_distance = max(0, min(100, norm_distance))  # Ограничиваем
        
        # Set volume
        sound_volume = norm_distance / 100.0
        volume.SetMasterVolumeLevelScalar(sound_volume, None)

        # Цвет круга меняется от красного (<20%) до зелёного
        color = (0, 0, 255) if norm_distance < 20 else (0, 255, 0)

        # Рисуем средний круг
        cx, cy = (x4 + x8) // 2, (y4 + y8) // 2
        cv2.circle(img, (cx, cy), 8, color, cv2.FILLED)

        # Показываем нормализованное расстояние на экране
        cv2.putText(img, f"Norm Dist: {int(norm_distance)}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print(f"Raw dist: {dist_4_8:.2f}, Normalized: {norm_distance:.2f}%")
    else:
        calibrated = False


    # FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f"FPS: {int(fps)}", (25, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Result", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
