# Использование MediaPipe
MediaPipe Solutions предоставляет набор библиотек и инструментов, позволяющих быстро применять методы искусственного интеллекта (ИИ) и машинного обучения (МО) в ваших приложениях. Вы можете сразу подключить эти решения к своим приложениям, настроить их в соответствии со своими потребностями и использовать на нескольких платформах разработки.
Официальный сайт [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ru)  

### Установка необходимого и первый запуск
##### 1. Python 3.9-3.12
- Скачать можно на официальном сайте [python.org](https://www.python.org/downloads/windows/).  
- Самые новые версии могут временно не поддерживаться YOLO.  
- Прямая ссылка на [Python 3.12.10](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) (100% рабочая)

##### 2. Установить нужные библиотеки
- в командную строку написать: pip3 install mediapipe opencv-python

##### 3. Запустить тестовый скрип
Запустить python IDLE -> file -> new file  
Написать код и сохранить в любую папку  
Код находит и отслеживает положение рук и пальце. [Полная статья](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)

```
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,             # 0/1/2: сложность модели. Выше — точнее, но медленнее (по умолчанию 1)
    max_num_hands=4,                # максимум обнаруживаемых рук в кадре
    min_detection_confidence=0.5,   # порог уверенности для первичного детекта рук (0..1)
    min_tracking_confidence=0.5     # порог для продолжения трекинга; ниже — вернётся к детекту (0..1)
) as hands:
    while cv2.waitKey(1) != 27:
        
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        start_time = time.perf_counter()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm, hness in zip(res.multi_hand_landmarks, res.multi_handedness):
                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )
        
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Press 'Esc' to exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hands", frame)

cap.release()
cv2.destroyAllWindows()

```
