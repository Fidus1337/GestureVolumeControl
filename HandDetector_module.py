import cv2
import mediapipe as mp

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
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def detect_hands(self, img, draw_hands=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands_detector.process(img_RGB)

        if draw_hands and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNumber=0):
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if handNumber < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[handNumber]
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list

# Testing the module
def main():
    my_detector = HandsDetector()  # Create instance of the hand detector
    cap = cv2.VideoCapture(0)      # Open default webcam

    ptime = 0  # Previous time for FPS calculation

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        if not success:
            break

        img = my_detector.detect_hands(img)        # Detect and draw hand landmarks
        lm_list = my_detector.findPositions(img)   # Get positions of landmarks

        if lm_list:
            # Example: draw a green circle at the tip of the index finger (landmark ID 8)
            cv2.circle(img, (lm_list[8][1], lm_list[8][2]), 10, (0, 255, 0), cv2.FILLED)

        # Calculate frames per second (FPS)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display the FPS on the image
        cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Show the image with the results
        cv2.imshow("Result", img)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
