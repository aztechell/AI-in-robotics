# Использование Ultralytics YOLO
Ultralytics YOLO — это открытая библиотека на Python для обучения и применения моделей YOLO предназначенных для задач компьютерного зрения: обнаружение объектов, сегментация, классификация.
Официальный сайт [Ultralytics YOLO.](https://docs.ultralytics.com/ru/)  

### Установка необходимого и первый запуск
##### 1. Python 3.10 и выше
- Скачать можно на официальном сайте [python.org](https://www.python.org/downloads/windows/).  
- Самые новые версии могут временно не поддерживаться YOLO.  
- Прямая ссылка на [Python 3.13.2](https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe) (100% рабочая)

##### 2. Установить драйверы NVIDIA (Только если есть видеокарта NVIDIA RTX)

- CUDA Toolkit [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
-  cuDNN [https://developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)

Выбирать версии для Windows 11.

##### 3. Установить pytorch
- Выбрать нужную версию [здесь](https://pytorch.org/get-started/locally/) и установить через командную строку. (Win + R, cmd)  
Например: pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu129](https://download.pytorch.org/whl/cu129)

##### 4. Установить нужные библиотеки
- в командную строку написать: pip3 install ultralytics opencv-python

##### 5. Скачать предобученную модель
- для первого примера скачать YOLO11n из [https://docs.ultralytics.com/ru/tasks/detect/](https://docs.ultralytics.com/ru/tasks/detect/)
Больше моделей и датасетов [https://docs.ultralytics.com/ru/datasets/](https://docs.ultralytics.com/ru/datasets/))

##### 6. Запустить тестовый скрип
Запустить python IDLE -> file -> new file
Написать код и сохранить в папку со скаченной моделью YOLO11n
```
import cv2
from ultralytics import YOLO
import torch
import time

# Проверка используется gpu или cpu
device = 'cuda' if torch.cuda.is_available() else  'cpu'
print(f"Using device: {device}")

# Загружаем нужную модель
model = YOLO('yolo11n.pt').to(device)
print("Available classes:", model.names)

# включаем камеру
webcamera = cv2.VideoCapture(0)

while True:
    #проверяем что камера работает и получаем изображение
    success, frame = webcamera.read()
    if not success:
        break

    # засекаем время для подсчёта FPS
    start_time = time.time()

    # Применяем модель на кадр
    results = model.track(frame, conf=0.8, imgsz=640)

    # Подсчёт ФПС
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Отображение ФПС
    cv2.putText(frame, f"FPS: {int(fps)}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Press 'Esc' to exit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Вывод изображения
    cv2.imshow("Live Camera", results[0].plot())

    # Нажать Esc для закрытия приложения
    if cv2.waitKey(1) == 27:
        break

# Отключение камеры и закрытие окна
webcamera.release()
cv2.destroyAllWindows()
```

Для смены камеры меняем цифру в строке 
``` webcamera = cv2.VideoCapture(0) ```  

Python скрипт для проверки списка камер:
```
import cv2
for i in range(10):
    cap=cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok=cap.isOpened()
    cap.release()
    if ok: print(i) 
```
Либо через командную строку:  
```wmic path Win32_PnPEntity where "PNPClass='Camera' OR PNPClass='Image'" get Name,DeviceID```

##### Бонус:
Сокращённая версия кода
```
import cv2
from ultralytics import YOLO
model = YOLO('yolo11m-pose.pt').to('cuda')
wc = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    results = model.track(wc.read()[1], conf=0.8, imgsz=640, verbose=False)
    cv2.imshow("Live Camera", results[0].plot())

wc.release(); cv2.destroyAllWindows()
```

### YOLO v11 имеет разные тренированные модели

<img src="../img/img_3.png" alt="desc" width="700">

**Classify** — задача классификации изображений. Определяет один класс для всего изображения (или кропа).  
**Detect** — задача обнаружения объектов. Находит каждый объект и присваивает класс, обрамляя прямоугольным bbox.  
**Segment** — задача сегментации экземпляров. Находит каждый объект, присваивает класс и выдает пиксельную маску его контура (часто вместе с bbox).  
**Track** — задача трекинга объектов во времени. Присваивает устойчивые ID уже найденным объектам в видеопотоке, связывая их между кадрами.  
**Pose** — задача оценки позы. Находит объект (обычно человека) и предсказывает координаты ключевых точек скелета (часто вместе с bbox и классом).  
**OBB** — задача обнаружения с повернутыми рамками. Находит объект, присваивает класс и выдает ориентированный прямоугольник (bbox с углом).  

<table>
  <thead>
    <tr>
      <th>Модель</th>
      <th>Имена файлов</th>
      <th>Задача</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>YOLO11</td>
      <td>
        <code>yolo11n.pt</code>
        <code>yolo11s.pt</code>
        <code>yolo11m.pt</code>
        <code>yolo11l.pt</code>
        <code>yolo11x.pt</code>
      </td>
      <td><a href="https://docs.ultralytics.com/ru/tasks/detect/">Обнаружение</a></td>
    </tr>
    <tr>
      <td>YOLO11-seg</td>
      <td>
        <code>yolo11n-seg.pt</code>
        <code>yolo11s-seg.pt</code>
        <code>yolo11m-seg.pt</code>
        <code>yolo11l-seg.pt</code>
        <code>yolo11x-seg.pt</code>
      </td>
      <td><a href="https://docs.ultralytics.com/ru/tasks/segment/">Сегментация экземпляров</a></td>
    </tr>
    <tr>
      <td>YOLO11-pose</td>
      <td>
        <code>yolo11n-pose.pt</code>
        <code>yolo11s-pose.pt</code>
        <code>yolo11m-pose.pt</code>
        <code>yolo11l-pose.pt</code>
        <code>yolo11x-pose.pt</code>
      </td>
      <td><a href="https://docs.ultralytics.com/ru/tasks/pose/">Поза/Ключевые точки</a></td>
    </tr>
    <tr>
      <td>YOLO11-obb</td>
      <td>
        <code>yolo11n-obb.pt</code>
        <code>yolo11s-obb.pt</code>
        <code>yolo11m-obb.pt</code>
        <code>yolo11l-obb.pt</code>
        <code>yolo11x-obb.pt</code>
      </td>
      <td><a href="https://docs.ultralytics.com/ru/tasks/obb/">Ориентированное обнаружение</a></td>
    </tr>
    <tr>
      <td>YOLO11-cls</td>
      <td>
        <code>yolo11n-cls.pt</code>
        <code>yolo11s-cls.pt</code>
        <code>yolo11m-cls.pt</code>
        <code>yolo11l-cls.pt</code>
        <code>yolo11x-cls.pt</code>
      </td>
      <td><a href="https://docs.ultralytics.com/ru/tasks/classify/">Классификация</a></td>
    </tr>
  </tbody>
</table>

### Дообучение моделей

#### Датасет
##### Предобученные модели и собранные датасеты 
Сайты:  
- universe.roboflow.com  
- huggingface.co  
- github.com  

#### Разметка данных

#### Дообучение

### Получение координат
Для работы с данными полученными от ИИ обычно выводят центральную координату найденных объектов.

<details>
<summary>Выводим координаты X, Y</summary>

```
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Получаем координаты bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Находим центр

        class_id = int(box.cls[0])  #Получаем id
        class_name = model.names[class_id]  # Получаем название класса

        # Extract tracking ID (if available)
        track_id = int(box.id[0]) if box.id is not None else -1

        print(f"Object: {class_name} (ID: {track_id}) at X={center_x}, Y={center_y}")
```
</details>
  
### Примеры с Pose
В модели позы YOLO11 по умолчанию имеется 17 ключевых точек, каждая из которых представляет собой отдельную часть человеческого тела. Ниже приведена схема соответствия каждого индекса соответствующему суставу тела:

<details>
<summary>Список ключевых точек YOLO11-pose</summary>

0. Нос  
1. Левый глаз  
2. Правый глаз  
3. Левое ухо  
4. Правое ухо  
5. Левое плечо  
6. Правое плечо  
7. Левый локоть  
8. Правый локоть  
9. Левое запястье  
10. Правое запястье  
11. Левое бедро  
12. Правое бедро  
13. Левое колено  
14. Правое колено  
15. Левая лодыжка  
16. Правая лодыжка  

</details>

Для отдельного нахождения координат одной ключевой точки, нужно извлечь её через results[0].keypoints.xy(0, **номер точки**). Например, правое запястье (точка 10).

<details>
<summary>Одна точка</summary>

```
import cv2
from ultralytics import YOLO
import torch
import time

device = 'cuda' if torch.cuda.is_available() else  'cpu'
print(f"Using device: {device}")

model = YOLO('YOLO11n-pose.pt').to(device)
print("Available classes:", model.names)

webcamera = cv2.VideoCapture(0)

j = 10  # правое запястье

while cv2.waitKey(1) != 27:
    success, frame = webcamera.read()
    if not success:
        break

    start_time = time.time()
    results = model.track(frame, conf=0.8, imgsz=640, verbose=False)

    r = results[0]
    if r.keypoints is not None and r.keypoints.xy.size(0) > 0:
        k = r.keypoints.xy[0, j]
        x, y = int(k[0].item()), int(k[1].item())
        cv2.circle(frame, (x, y), 6, (0,255,255), -1)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(frame, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", frame)

webcamera.release()
cv2.destroyAllWindows()
```

</details>       

Нахождение нескольких точек.
<details>
<summary>Несколько точек</summary>

```
import cv2
from ultralytics import YOLO
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('YOLO11n-pose.pt').to(device)

cap = cv2.VideoCapture(0)

points = [0, 1, 2]  

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.time()
    results = model.track(frame, conf=0.8, imgsz=640, verbose=False)
    r = results[0]

    if r.keypoints is not None and r.keypoints.xy is not None:
        kpts = r.keypoints.xy
        np, nk = kpts.shape[0], kpts.shape[1]
        for pid in range(np):
            for idx in points:
                if idx >= nk:
                    continue
                x, y = map(int, kpts[pid, idx].tolist())
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

    fps = 1.0 / max(1e-6, time.time() - t0)
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera", frame)

cap.release()
cv2.destroyAllWindows()

```
</details>

Несколько точек с линиями.
<details>
<summary>Линии</summary>

```
import cv2
from ultralytics import YOLO
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('YOLO11n-pose.pt').to(device)

cap = cv2.VideoCapture(0)

pairs = [(1, 2), (5, 7), (7, 9), (6, 8), (8, 10)]  # индексы keypoints, которые соединяем

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.time()
    results = model.track(frame, conf=0.8, imgsz=640, verbose=False)
    r = results[0]

    if r.keypoints is not None and r.keypoints.xy is not None:
        kpts = r.keypoints.xy  # shape: (num_persons, num_kpts, 2)
        num_persons, num_kpts = kpts.shape[0], kpts.shape[1]

        for pid in range(num_persons):
            for a, b in pairs:
                if a >= num_kpts or b >= num_kpts:
                    continue
                x1, y1 = map(int, kpts[pid, a].tolist())
                x2, y2 = map(int, kpts[pid, b].tolist())
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                #точки:
                cv2.circle(frame, (x1, y1), 4, (0, 255, 255), -1)
                cv2.circle(frame, (x2, y2), 4, (0, 255, 255), -1)

    fps = 1.0 / max(1e-6, time.time() - t0)
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Camera", frame)

cap.release()
cv2.destroyAllWindows()

```
</details>

Пример игры.
<details>
<summary>Игра</summary>

```
import cv2, math
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('YOLO11n-pose.pt').to(device)

cap = cv2.VideoCapture(0)

target = 0          # нос
alpha = 0.05         # скорость подтягивания
conf_th = 0.5        # порог уверенности

cx = cy = tx = ty = None

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    if cx is None:
        cx = tx = w / 2
        cy = ty = h / 2

    r = model.track(frame, classes=0, conf=0.8, imgsz=640, verbose=False)[0]

    found = False
    if r.keypoints is not None and r.keypoints.xy is not None and r.keypoints.xy.size(0) > 0:
        kxy = r.keypoints.xy
        kcf = r.keypoints.conf  # может быть None
        num_kp = kxy.shape[1]

        if target < num_kp:
            x, y = float(kxy[0, target, 0]), float(kxy[0, target, 1])
            ok_conf = True if kcf is None else float(kcf[0, target]) >= conf_th
            if ok_conf and not (x != x or y != y):  # проверка на NaN
                tx, ty = x, y
                found = True

    # двигаем только если нашли новую цель
    if found:
        cx += (tx - cx) * alpha
        cy += (ty - cy) * alpha

    # рисуем
    if found:
        cv2.circle(frame, (int(tx), int(ty)), 4, (0, 255, 255), -1)  # цель
    cv2.circle(frame, (int(cx), int(cy)), 14, (0, 0, 255), 2)        # игрок

    # событие попадания
    if math.hypot(cx - tx, cy - ty) < 12:
        fs, thick = 3, 7
        (tw, th), _ = cv2.getTextSize("GAY", cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        org = ((w - tw) // 2, (h + th) // 2)
        cv2.putText(frame, "GAY", org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick, cv2.LINE_AA)

    cv2.imshow("Live Camera", frame)

cap.release()
cv2.destroyAllWindows()

```
</details>
<br>
<br>