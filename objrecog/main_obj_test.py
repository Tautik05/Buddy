import cv2
from objrecog.obj import ObjectDetector
from objrecog.perception import interpret_objects

detector = ObjectDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    labels = detector.detect(frame)
    response = interpret_objects(labels)

    if response:
        print(response)

    cv2.imshow("BUDDY Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
