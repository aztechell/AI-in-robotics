@echo off
echo === Deleting Ultralytics settings.json cache...

set "ULTRA_CONFIG=%APPDATA%\Ultralytics\settings.json"

if exist "%ULTRA_CONFIG%" (
    del "%ULTRA_CONFIG%"
    echo Deleted: %ULTRA_CONFIG%
) else (
    echo No cache found. Skipping deletion.
)

echo.
echo === Updating Python packages...

pip install -U ultralytics opencv-python

echo.
echo === Update complete. Press any key to exit.
pause >nul
