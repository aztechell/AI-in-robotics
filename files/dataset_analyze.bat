@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Usage: countfiles.bat [folder]
set "folder=%~1"
if not defined folder set "folder=."

pushd "%folder%" || (echo Error: folder not found & pause & exit /b 1)

rem Counts
for /f %%N in ('dir /a:-d /b "*.txt" 2^>nul ^| find /c /v ""') do set "txtcount=%%N"
for /f %%N in ('dir /a:-d /b "*.jpg" 2^>nul ^| find /c /v ""') do set "jpgcount=%%N"
for /f %%N in ('dir /a:-d /b "*.jpeg" 2^>nul ^| find /c /v ""') do set "jpegcount=%%N"
for /f %%N in ('dir /a:-d /b "*.png" 2^>nul ^| find /c /v ""') do set "pngcount=%%N"
for /f %%N in ('dir /a:-d /b "*.bmp" 2^>nul ^| find /c /v ""') do set "bmpcount=%%N"
for /f %%N in ('dir /a:-d /b "*.gif" 2^>nul ^| find /c /v ""') do set "gifcount=%%N"

set /a imgcount=jpgcount+jpegcount+pngcount+bmpcount+gifcount

rem Paired = txt whose basename has any image
set "paired=0"
for %%F in ("*.txt") do (
    set "base=%%~nF"
    if exist "!base!.jpg" (
        set /a paired+=1
    ) else if exist "!base!.jpeg" (
        set /a paired+=1
    ) else if exist "!base!.png" (
        set /a paired+=1
    ) else if exist "!base!.bmp" (
        set /a paired+=1
    ) else if exist "!base!.gif" (
        set /a paired+=1
    )
)

echo In "%folder%":
echo   TXT files                    : %txtcount%
echo   Images (JPG/JPEG/PNG/BMP/GIF): %imgcount%
echo   Paired image+txt (same name) : %paired%

popd
pause
exit /b 0
