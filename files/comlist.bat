@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo Serial ports (fast, no PowerShell)
echo PORT    VID:PID    MFR                NAME

rem 1) Список COM из SERIALCOMM
for /f "tokens=3" %%P in ('
  reg query HKLM\HARDWARE\DEVICEMAP\SERIALCOMM 2^>nul ^| findstr /R /C:"REG_SZ"
') do (
  set "PORT=%%P"
  call :DETAILS "%%P"
)

echo.
pause
exit /b

:DETAILS
set "PORT=%~1"
set "VP=" & set "MFR=" & set "NAME=" & set "BASE="

rem 2) Быстрый точечный поиск ключа с PortName == PORT
for /f "delims=" %%K in ('
  reg query "HKLM\SYSTEM\CurrentControlSet\Enum" /s /f "%PORT%" /e 2^>nul ^| findstr /I /R "\\Device Parameters$"
') do (
  set "BASE=%%K"
  set "BASE=!BASE:\Device Parameters=!"
  goto :HAVEBASE
)

:HAVEBASE
if defined BASE (
  for /f "tokens=2,*" %%a in ('reg query "!BASE!" /v FriendlyName 2^>nul ^| findstr /I "FriendlyName"') do set "NAME=%%b"
  if not defined NAME for /f "tokens=2,*" %%a in ('reg query "!BASE!" /v DeviceDesc 2^>nul ^| findstr /I "DeviceDesc"') do set "NAME=%%b"
  for /f "tokens=2,*" %%a in ('reg query "!BASE!" /v Mfg 2^>nul ^| findstr /I "Mfg"') do set "MFR=%%b"
  if not defined MFR for /f "tokens=2,*" %%a in ('reg query "!BASE!" /v Manufacturer 2^>nul ^| findstr /I "Manufacturer"') do set "MFR=%%b"
  
  rem Очистка INF-строк вида @oemXX.inf,%...%;Текст
  if "!MFR:~0,1!"=="@" for /f "tokens=2* delims=;" %%s in ("!MFR!") do set "MFR=%%s"
  if "!NAME:~0,1!"=="@" for /f "tokens=2* delims=;" %%s in ("!NAME!") do set "NAME=%%s"

  for /f "delims=" %%I in ('reg query "!BASE!" /v HardwareID 2^>nul ^| findstr /I "VID_"') do (
    set "LINE=%%I"
    set "TMP=!LINE:*VID_=!"
    set "VID=!TMP:~0,4!"
    set "TMP=!LINE:*PID_=!"
    set "PID=!TMP:~0,4!"
    if defined VID if defined PID set "VP=[!VID!:!PID!]"
    goto :PRINT
  )
)

:PRINT
echo %PORT%  !VP!  !MFR!  !NAME!
goto :eof
