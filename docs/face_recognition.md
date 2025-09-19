# Распознавание лиц
Распознавание лиц – это биометрическая технология, которая позволяет идентифицировать или подтвердить личность человека по особенностям его лица, используя алгоритмы искусственного интеллекта и машинного обучения.  
## Логика работы:  
1. Создаётся база данных известных лиц. Обычно это папка с фотографиями по типу:   
./database/person1/1.jpg, 2.jpg   
./database/person2/1.jpg, 2.jpg     
2. Детектор лиц (mediapipe, mtcnn и тд.) находит лицо
3. Лицо преобразуется в векторное представление (embedding)
4. Преобразованное лицо отправляется в модель распознания лица (sface, ArcFace и тд.)
5. Модель сравнивает полученное лицо с лицами из базы данных. Обычно через метрику [Cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Выходное значение от -1 до 1. Чем выше значение, тем лица похожее
6. Если значение точности выше порога, то человек распознан

## НЕ рабочие варианты:  
- Библиотека InsightFace - проблемы с совместимостью версий зависимостей, плохо работает   
- Библиотека DeepFace - кривая сборка разных тяжелых моделей, работает плохо   
- Библиотека [face_recognition](https://github.com/ageitgey/face_recognition) - не работает на Windows   

## Рабочий вариант
### mediapipe + sface
[Mediapipe](mediapipe.md) как детектор лиц.  
SFace — это модель распознавания лиц, доступная в библиотеке OpenCV. Это модель, основанная на глубоком обучении, разработанная для эффективного и точного распознавания лиц.

1. установить python (не выше 3.12) и библиотеки 
   > pip install opencv-contrib-python mediapipe numpy
2. Скачать модель [face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) и загрузить в папку, где будет код. (Оригинальный источник [Зоопарк моделей OpenCV](https://github.com/opencv/opencv_zoo))
3. Создать папку для базы данных, например db. Внутри неё создать папки для каждого человека и загрузить туда фотографии. Пример:

    ```
    db/
    ├─ Azat/
    │  └─ 1.jpg
    ├─ Diana
    │  ├─ lol.jpg
    │  └─ kek.png
    └─ Obama
       ├─ obama.jpg
       └─ barak.png
    ```

4. Пишем код:

<details>
<summary>Face recognition</summary>

```
import os, time, cv2, numpy as np, mediapipe as mp
from pathlib import Path

DB_PATH    = "./db"   
SFACE_ONNX = "./face_recognition_sface_2021dec.onnx"
THRESH     = 0.363  

recognizer = cv2.FaceRecognizerSF_create(SFACE_ONNX, "")
mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                                       max_num_faces=10,
                                                                       refine_landmarks=True,
                                                                       min_detection_confidence=0.5,
                                                                       min_tracking_confidence=0.5)

def list_images(root):        #создаем список фото
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for p in d.rglob("*"):
                if p.suffix.lower() in exts:
                    yield d.name, str(p)

def mesh_5pts(lm, w, h):   #facemesh переводим в sface
    L = lm.landmark
    def px(i): return (int(L[i].x*w), int(L[i].y*h))

    re = tuple(np.mean([px(33), px(133)], axis=0).astype(int))    # right eye center
    le = tuple(np.mean([px(263), px(362)], axis=0).astype(int))  # left eye center
    nt = px(1)                                                                              # nose tip
    rmc = px(61)                                                                         # right mouth corner
    lmc = px(291)                                                                       # left mouth corner
    
    xs = [re[0], le[0], nt[0], rmc[0], lmc[0]]
    ys = [re[1], le[1], nt[1], rmc[1], lmc[1]]                                  #координаты 5 точек
    x, y = max(0, min(xs)), max(0, min(ys))                              #находим крайние точки
    wbb, hbb = max(xs)-x+1, max(ys)-y+1                               #находим ширину и высоту 
    cx, cy = x + wbb//2, y + hbb//2                                              #находим центр                                              
    scale = 1.8                                                                           
    nw, nh = int(wbb*scale), int(hbb*scale)                              #увеличиваем масштаб коробки
    x = max(0, cx - nw//2)
    y = max(0, cy - nh//2)
    x2 = min(w-1, x + nw)
    y2 = min(h-1, y + nh)
    wbb, hbb = x2 - x, y2 - y

    row = np.array([[x, y, wbb, hbb,
                     re[0], re[1], le[0], le[1], nt[0], nt[1],
                     rmc[0], rmc[1], lmc[0], lmc[1]]], dtype=np.float32)
    return row, (x, y, wbb, hbb)

def embed_from_img(img):    #преобразуем кадр в векторное предствление
    h, w = img.shape[:2]
    res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:
            row, (x,y,wb,hb) = mesh_5pts(lm, w, h)
            try:
                face = recognizer.alignCrop(img, row)
            except Exception:
                x2, y2 = min(w-1, x+wb), min(h-1, y+hb)
                face = img[y:y2, x:x2].copy()
            feat = recognizer.feature(face)
            out.append((feat, (x,y,wb,hb)))
    return out

gallery = {}
for label, path in list_images(DB_PATH):     #создаём галерею из фото в базе данных
    img = cv2.imread(path)
    if img is None: continue
    fb = embed_from_img(img)
    if not fb: continue
    feat, _ = fb[0]
    gallery.setdefault(label, []).append(feat)

labels = sorted(gallery.keys())
G = []
for k in labels:
    M = np.mean(np.vstack(gallery[k]), axis=0)
    M = M / max(np.linalg.norm(M), 1e-9)
    G.append(M)
G = np.vstack(G).astype(np.float32)  # [N,512]


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("MP+SFace", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MP+SFace", 640, 480)
ema_fps = 0.0

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()

    fb = embed_from_img(frame)
    for feat, (x,y,wb,hb) in fb:                                             #сравниваем лица
        q = feat / max(np.linalg.norm(feat), 1e-9)
        sims = (G @ q.ravel()).astype(np.float32)  # cosine similarity
        i = int(np.argmax(sims))
        s = float(sims[i])
        
        name = labels[i] if s >= THRESH else "unknown"    #пишем имя если нашли
        cv2.rectangle(frame, (x,y), (x+wb, y+hb), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({s:.2f})", (x, max(0,y-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    dt = max(time.perf_counter() - t0, 1e-6)
    ema_fps = (ema_fps*0.9 + 0.1*(1.0/dt)) if ema_fps else 1.0/dt   #Считаем фпс
    
    cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)  #Выводим фпс
    
    cv2.imshow("MP+SFace", frame) #вывод изображения

cap.release(); cv2.destroyAllWindows()

```

</details>
<br>

### Чистый OpenCV (YuNet + sface)
YuNet — это детектор лиц на основе сверточной нейронной сети (CNN), известный своей высокой эффективностью и точностью, особенно при распознавании лиц под разными углами и с перекрытиями.   
SFace — это модель распознавания лиц, доступная в библиотеке OpenCV. Это модель, основанная на глубоком обучении, разработанная для эффективного и точного распознавания лиц.  
Обе модели являются частью библиотеки OpenCV, поэтому отлично работают вместе.   

1. установить python (не выше 3.13) и библиотеки 
   > pip install opencv-contrib-python numpy
2. Скачать модель [face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) и [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) и загрузить в папку, где будет код. (Оригинальный источник [Зоопарк моделей OpenCV](https://github.com/opencv/opencv_zoo))
3. Создать папку для базы данных, например db. Внутри неё создать папки для каждого человека и загрузить туда фотографии. Пример:

    ```
    db/
    ├─ Azat/
    │  └─ 1.jpg
    ├─ Diana
    │  ├─ lol.jpg
    │  └─ kek.png
    └─ Obama
       ├─ obama.jpg
       └─ barak.png
    ```

4. Пишем код:

<details>
<summary>Face recognition</summary>

```
import os, time, cv2, numpy as np
from pathlib import Path

DB_PATH     = "./db"
SFACE_ONNX  = "./face_recognition_sface_2021dec.onnx"      
YUNET_ONNX  = "./face_detection_yunet_2023mar.onnx"
THRESH      = 0.363

recognizer = cv2.FaceRecognizerSF_create(SFACE_ONNX, "")
detector   = cv2.FaceDetectorYN_create(YUNET_ONNX, "", (320,320), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

def list_images(root):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for p in d.rglob("*"):
                if p.suffix.lower() in exts:
                    yield d.name, str(p)

def embed_from_img(img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    ok, faces = detector.detect(img)
    out = []
    if ok and faces is not None and len(faces):
        for f in faces:
            x, y, wb, hb = f[:4].astype(int)
            row = f[:14].astype(np.float32).reshape(1, 14)  # box + 5 keypoints
            face = recognizer.alignCrop(img, row)
            feat = recognizer.feature(face)
            out.append((feat, (x, y, wb, hb)))
    return out

gallery = {}
for label, path in list_images(DB_PATH):
    img = cv2.imread(path)
    if img is None: continue
    fb = embed_from_img(img)
    if not fb: continue
    feat, _ = fb[0]                       # largest/first face per image
    gallery.setdefault(label, []).append(feat)

labels = sorted(gallery.keys())
G = []
for k in labels:
    M = np.mean(np.vstack(gallery[k]), axis=0)
    M = M / max(np.linalg.norm(M), 1e-9)
    G.append(M)
G = np.vstack(G).astype(np.float32)      

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("YuNet+SFace", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YuNet+SFace", 640, 480)
ema_fps = 0.0

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()

    fb = embed_from_img(frame)
    for feat, (x,y,wb,hb) in fb:
        q = feat / max(np.linalg.norm(feat), 1e-9)
        sims = (G @ q.ravel()).astype(np.float32)   # cosine
        i = int(np.argmax(sims)); s = float(sims[i])
        
        name = labels[i] if s >= THRESH else "unknown"
        cv2.rectangle(frame, (x,y), (x+wb, y+hb), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({s:.2f})", (x, max(0,y-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    dt = max(time.perf_counter() - t0, 1e-6)
    ema_fps = (ema_fps*0.9 + 0.1*(1.0/dt)) if ema_fps else 1.0/dt
    cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.imshow("YuNet+SFace", frame)

cap.release(); cv2.destroyAllWindows()

```

</details>
<br>