import cv2
import mediapipe as mp
import time

class HandsDetector():
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpHands = mp.solutions.hands
        self.hands_detector = self.mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

    def detect_hands(self, img, draw_hands=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands_detector.process(img_RGB)

        if draw_hands and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNumber=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list


def main():
    my_detector = HandsDetector()
    cap = cv2.VideoCapture(0)

    ptime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = my_detector.detect_hands(img)
        lm_list = my_detector.findPositions(img)

        if lm_list:
            # Пример: рисуем на кончике указательного пальца (id=8)
            cv2.circle(img, (lm_list[8][1], lm_list[8][2]), 10, (0, 255, 0), cv2.FILLED)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
