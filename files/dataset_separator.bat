@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Usage: organize_dataset.bat [count_folder] [dataset_root]
set "folder=%~1"
if not defined folder set "folder=."
set "ROOT=%~2"
if not defined ROOT set "ROOT=dataset"

pushd "%folder%" || (echo Error: folder not found & pause & exit /b 1)

rem --- init counts
for %%V in (txtcount jpgcount jpegcount pngcount bmpcount gifcount imgcount paired) do set "%%V=0"

for /f %%N in ('dir /a:-d /b "*.txt"  2^>nul ^| find /c /v ""') do set "txtcount=%%N"
for /f %%N in ('dir /a:-d /b "*.jpg"  2^>nul ^| find /c /v ""') do set "jpgcount=%%N"
for /f %%N in ('dir /a:-d /b "*.jpeg" 2^>nul ^| find /c /v ""') do set "jpegcount=%%N"
for /f %%N in ('dir /a:-d /b "*.png"  2^>nul ^| find /c /v ""') do set "pngcount=%%N"
for /f %%N in ('dir /a:-d /b "*.bmp"  2^>nul ^| find /c /v ""') do set "bmpcount=%%N"
for /f %%N in ('dir /a:-d /b "*.gif"  2^>nul ^| find /c /v ""') do set "gifcount=%%N"

set /a imgcount=jpgcount+jpegcount+pngcount+bmpcount+gifcount

rem --- paired detection: save basenames into list
set "paired=0"
>_pairs.tmp (
  for %%F in ("*.txt") do (
      set "base=%%~nF"
      set "ext="
      if exist "!base!.jpg"  set "ext=jpg"
      if exist "!base!.jpeg" set "ext=jpeg"
      if exist "!base!.png"  set "ext=png"
      if exist "!base!.bmp"  set "ext=bmp"
      if exist "!base!.gif"  set "ext=gif"
      if defined ext (
          echo !base!;!ext!
          set /a paired+=1
      )
  )
)

rem --- 80/20 split (integer)
set /a trainPairs=paired*80/100
set /a valPairs=paired-trainPairs

echo In "%folder%":
echo   TXT files                     : %txtcount%
echo   Images (JPG/JPEG/PNG/BMP/GIF) : %imgcount%
echo   Paired image+txt (same name)  : %paired%
echo   Split 80/20 on pairs          : train %trainPairs%, validation %valPairs%
echo.

choice /c YN /n /m "Create YOLO dataset folder and move files? [Y/N]"
if errorlevel 2 goto :done

rem --- create dataset folders
for %%D in (
  "%ROOT%\images\train"
  "%ROOT%\images\val"
  "%ROOT%\labels\train"
  "%ROOT%\labels\val"
) do if not exist "%%~D" mkdir "%%~D"

rem --- dataset.yaml
>"%ROOT%\dataset.yaml" echo train: images/train
>>"%ROOT%\dataset.yaml" echo val: images/val
>>"%ROOT%\dataset.yaml" echo nc: 3
>>"%ROOT%\dataset.yaml" echo names: ['person', 'car', 'dog']

>"%ROOT%\train.py" echo from ultralytics import YOLO
>>"%ROOT%\train.py" echo model = YOLO("yolo11n.pt")
>>"%ROOT%\train.py" echo model.train(
>>"%ROOT%\train.py" echo     data="dataset.yaml",
>>"%ROOT%\train.py" echo     epochs=50,
>>"%ROOT%\train.py" echo     imgsz=640,
>>"%ROOT%\train.py" echo     batch=16,
>>"%ROOT%\train.py" echo     workers=4,
>>"%ROOT%\train.py" echo     project="yolo11_train",
>>"%ROOT%\train.py" echo     name="exp1",
>>"%ROOT%\train.py" echo     exist_ok=True
>>"%ROOT%\train.py" echo )

set /a train=0, val=0
for /f "usebackq tokens=1,2 delims=;" %%A in ("_pairs.tmp") do (
    rem 1) взять новое случайное число для каждой итерации
    call set "r=%%random%%"
    rem 2) привести к 0..99
    set /a r=r%%100

    if !train! lss %trainPairs% if !val! lss %valPairs% (
        if !r! lss 80 (
            move "%%A.%%B" "%ROOT%\images\train\" >nul
            move "%%A.txt" "%ROOT%\labels\train\" >nul
            set /a train+=1
        ) else (
            move "%%A.%%B" "%ROOT%\images\val\" >nul
            move "%%A.txt" "%ROOT%\labels\val\" >nul
            set /a val+=1
        )
    ) else if !train! lss %trainPairs% (
        rem train quota not reached, send remainder to train
        move "%%A.%%B" "%ROOT%\images\train\" >nul
        move "%%A.txt" "%ROOT%\labels\train\" >nul
        set /a train+=1
    ) else (
        rem val quota not reached, send remainder to val
        move "%%A.%%B" "%ROOT%\images\val\" >nul
        move "%%A.txt" "%ROOT%\labels\val\" >nul
        set /a val+=1
    )
)

endlocal

echo Files moved and dataset.yaml created.

:done
del _pairs.tmp 2>nul
popd
pause
exit /b 0
