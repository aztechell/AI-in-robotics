@echo off
setlocal
set "ROOT=dataset"

if not exist "%ROOT%" mkdir "%ROOT%"

for %%D in (
  "%ROOT%\images\train"
  "%ROOT%\images\val"
  "%ROOT%\images\test"
  "%ROOT%\labels\train"
  "%ROOT%\labels\val"
  "%ROOT%\labels\test"
) do if not exist "%%~D" mkdir "%%~D"

rem dataset.yaml
>"%ROOT%\dataset.yaml" echo train: ../images/train
>>"%ROOT%\dataset.yaml" echo val: ../images/val
>>"%ROOT%\dataset.yaml" echo test: images/test
>>"%ROOT%\dataset.yaml" echo.
>>"%ROOT%\dataset.yaml" echo nc: 3  # количество классов
>>"%ROOT%\dataset.yaml" echo names: ['person', 'car', 'dog']

rem train.py
>"%ROOT%\train.py" echo from ultralytics import YOLO
>>"%ROOT%\train.py" echo.
>>"%ROOT%\train.py" echo model = YOLO("yolo11n.pt")  # выбираем модель для дообучения
>>"%ROOT%\train.py" echo.
>>"%ROOT%\train.py" echo # Train the model
>>"%ROOT%\train.py" echo model.train(
>>"%ROOT%\train.py" echo     data^=^"dataset.yaml^",       # путь к dataset.yaml 
>>"%ROOT%\train.py" echo     epochs^=50,                   # количество эпох(кругов) обучения
>>"%ROOT%\train.py" echo     imgsz^=640,                   # размер картинок
>>"%ROOT%\train.py" echo     batch^=16,                    # количество картинок за один шаг
>>"%ROOT%\train.py" echo     workers^=4,                   # число параллельных процессов,
>>"%ROOT%\train.py" echo     project^=^"yolo11_train^",    # папка для сохранения результатов
>>"%ROOT%\train.py" echo     name^=^"exp1^",               # название эксперимента
>>"%ROOT%\train.py" echo     exist_ok^=True                # переписать если есть такая папка
>>"%ROOT%\train.py" echo ^)

echo Done.
