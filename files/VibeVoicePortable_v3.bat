@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
set "INSTALL_DIR=%~dp0"
set "PYTHON_DIR=%INSTALL_DIR%python_embedded"
set "FFMPEG_DIR=%INSTALL_DIR%ffmpeg"
set "VOICEPACK_DIR=%INSTALL_DIR%voice-pack"
set "VENV_DIR=%INSTALL_DIR%venv"
set "GIT_DIR=%INSTALL_DIR%PortableGit"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip"
set "FFMPEG_URL=https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/ffmpeg.zip"
set "VOICEPACK_URL=https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip"
set "GRADIODEMO_URL=https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/gradio_demo.py"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/PortableGit-2.44.0-64-bit.7z.exe"
set "REPO_URL=https://github.com/rsxdalv/VibeVoice"
set "REPO_DIR=%INSTALL_DIR%VibeVoice_repo"

REM Setup clean PIP cache while re/installing
set CLEAN_PIP_CACHE=0
REM Setup clean HugginFace models cache while re/installing
set CLEAN_HF_HOME=0
REM Setup clean VoicePack cache while re/installing
set CLEAN_VOICEPACK=1
REM Setup clean ffmpeg folder while re/installing
set CLEAN_FFMPEG=0
REM General directory for models, not a specific model path

set "MODELS_DIR=%INSTALL_DIR%models"

REM Setup local directories for cache and temp files
set "HF_HOME=%INSTALL_DIR%huggingface_cache"
set "PIP_CACHE_DIR=%INSTALL_DIR%cache"
set "PYTHONUSERBASE=%INSTALL_DIR%python_user_base"
set "TEMP=%INSTALL_DIR%temp"
set "TMP=%INSTALL_DIR%temp"
set "MAX_RETRIES=3"
set "RETRY_DELAY=5"

:main_menu
cls
echo ======================================
echo          VibeVoice Main Menu
echo ======================================
echo 1. Start VibeVoice Normal Mode
echo 2. Start VibeVoice CPU Mode (Low VRAM)
echo 3. Install/Reinstall VibeVoice
echo ======================================
echo By @li_aeron and @nerual_dreming, 2025
echo https://t.me/neuroport
echo ======================================
set /p choice=Choose action 1-3:
if "%choice%"=="1" goto start_vibe
if "%choice%"=="2" goto start_vibe_low
if "%choice%"=="3" goto install_vibe
echo Wrong choice. Please, try again.
pause
goto main_menu

:install_vibe
cls
echo ===================================================================================
echo                     Install/Reinstall VibeVoice
echo ===================================================================================
echo 1. Install/Reinstall with Torch 2.*.* CUDA 12.1 (GTX10x-16x, RTX20x-40x)
echo 2. Install/Reinstall with Torch 2.*.* CUDA 12.1 + Flash Attention (RTX20x-40x ONLY)
echo 3. Install/Reinstall with Torch 2.7.1 CUDA 12.8 (GTX10x-16x, RTX20x-50x)
echo 4. Back
echo ===================================================================================
set /p file_choice=Choose action 1-4: 
if "%file_choice%"=="1" goto install_vibe_cu121
if "%file_choice%"=="2" goto install_vibe_cu121flashattn
if "%file_choice%"=="3" goto install_vibe_271cu128
if "%file_choice%"=="4" goto main_menu
echo Wrong choice. please, try again.
pause
goto install_vibe

:install_vibe_cu121
cls
REM Step 1: Clean old installation if exists
echo [1/10] Preparing for installation: cleaning up old files...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%REPO_DIR%" rmdir /s /q "%REPO_DIR%"
if exist "%PYTHONUSERBASE%" rmdir /s /q "%PYTHONUSERBASE%"
if exist "%TEMP%" rmdir /s /q "%TEMP%"
if exist "%GIT_DIR%" rmdir /s /q "%GIT_DIR%"
if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
if "%CLEAN_PIP_CACHE%"=="1" (
    if exist "%PIP_CACHE_DIR%" rmdir /s /q "%PIP_CACHE_DIR%"
)

if "%CLEAN_HF_HOME%"=="1" (
    if exist "%HF_HOME%" rmdir /s /q "%HF_HOME%"
)

if "%CLEAN_VOICEPACK%"=="1" (
    if exist "%VOICEPACK_DIR%" rmdir /s /q "%VOICEPACK_DIR%"
)

if "%CLEAN_FFMPEG%"=="1" (
    if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%"
)
mkdir "%TEMP%"
echo Done.

REM Step 2: Download and install portable Git
echo.
echo [2/10] Downloading and installing portable Git...

set "EXPECTED_HASH=1FC64CA91B9B475AB0ADA72C9F7B3ADDBE69A6C8F520BE31425CF21841CCA369"
set "TMP_FILE=%INSTALL_DIR%PortableGit.tmp"
set "FINAL_FILE=%INSTALL_DIR%PortableGit.exe"

if not exist "%GIT_DIR%" (
    if exist "%FINAL_FILE%" (
        call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
        if "%HASH_MATCH%"=="1" (
            echo Git installer already downloaded and verified.
            goto InstallGit
        ) else (
            echo Hash mismatch. Removing corrupted file.
            del "%FINAL_FILE%"
        )
    )

    if exist "%TMP_FILE%" (
        echo Resuming Git download...
    ) else (
        echo Starting Git download...
    )

    curl -C - --retry %MAX_RETRIES% --retry-delay 2 -L "%GIT_URL%" -o "%TMP_FILE%"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Git after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    move /Y "%TMP_FILE%" "%FINAL_FILE%"

    call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
    if "%HASH_MATCH%"=="0" (
        echo Error: Hash mismatch after download.
        del "%FINAL_FILE%"
        del "%TMP_FILE%"
        pause
        exit /b 1
    )
    goto InstallGit
)

:InstallGit
"%FINAL_FILE%" -y -gm2 -nr -o"%GIT_DIR%"
del "%FINAL_FILE%"

set "PATH=%GIT_DIR%\bin;%~dp0python_embedded;%~dp0ffmpeg\bin;%PATH%"
echo Done.

REM Step 3: Download and extract portable Python
echo.
echo [3/10] Downloading and setting up portable Python 3.11...
if not exist "%PYTHON_DIR%" (
    mkdir "%PYTHON_DIR%"
    call :DownloadFile "%PYTHON_URL%" "%INSTALL_DIR%python.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Python after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%python.zip' -DestinationPath '%PYTHON_DIR%'"
    del "%INSTALL_DIR%python.zip"

    echo.
    echo Configuring Python to use pip...
    powershell -Command "(Get-Content '%PYTHON_DIR%\python311._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python311._pth'"
    
    echo.
    echo Installing pip...
    call :DownloadFile "%GET_PIP_URL%" "%PYTHON_DIR%\get-pip.py"
     if %errorlevel% neq 0 (
        echo Error: Failed to download get-pip.py after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
)
echo Done.

REM Step 4: Download and extract ffmpeg
echo.
echo [4/10] Downloading and unpack ffmpeg binaries...
if not exist "%FFMPEG_DIR%" (
    mkdir "%FFMPEG_DIR%"
    call :DownloadFile "%FFMPEG_URL%" "%INSTALL_DIR%ffmpeg.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download FFMPEG after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%'"
    del "%INSTALL_DIR%ffmpeg.zip"

    echo.
)
echo Done.

REM Step 5: Download and extract VoicePack
echo.
echo [5/10] Downloading and unpack VoicePack...
if not exist "%VOICEPACK_DIR%" (
    mkdir "%VOICEPACK_DIR%"
    call :DownloadFile "%VOICEPACK_URL%" "%INSTALL_DIR%voice-pack.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download VoicePack after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%voice-pack.zip' -DestinationPath '%VOICEPACK_DIR%'"
    del "%INSTALL_DIR%voice-pack.zip"

    echo.
)
echo Done.

REM Step 6: Download and extract GradioDemo
echo.
echo [6/10] Downloading and unpack GradioDemo...
    call :DownloadFile "%GRADIODEMO_URL%" "%INSTALL_DIR%gradio_demo.py"
    if %errorlevel% neq 0 (
        echo Error: Failed to download GradioDemo after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    echo.
echo Done.

REM Step 7: Install virtualenv
echo.
echo [7/10] Installing virtualenv...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --no-cache-dir virtualenv
if %errorlevel% neq 0 (
    echo Error: Failed to install virtualenv after %MAX_RETRIES% attempts.
    pause
    exit /b 1
)
echo Done.

REM Step 8: Create virtual environment and install dependencies
echo.
echo [8/10] Cloning repository and installing dependencies...
if not exist "%REPO_DIR%" (
    call :RunCommandWithRetry "%GIT_DIR%\bin\git.exe" clone --depth 1 "%REPO_URL%" "%REPO_DIR%"
    if %errorlevel% neq 0 (
        echo Error: Failed to clone the VibeVoice repository after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
)
"%PYTHON_DIR%\python.exe" -m virtualenv --no-download "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo Error: Failed to create the virtual environment.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate"

echo Upgrading pip...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --cache-dir "%PIP_CACHE_DIR%" --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

echo Checking for uv in virtual environment...
set "UV_EXE=%VENV_DIR%\Scripts\uv.exe"

if not exist "%UV_EXE%" (
    echo 'uv' not found in venv. Attempting installation...

    REM Ensure pip is available in venv
    "%VENV_DIR%\Scripts\python.exe" -m ensurepip --default-pip >nul 2>&1
    if errorlevel 1 (
        echo Failed to bootstrap pip in venv. Trying get-pip.py...
        curl -L -o "%TEMP%\get-pip.py" "%GET_PIP_URL%"
        "%VENV_DIR%\Scripts\python.exe" "%TEMP%\get-pip.py"
        if errorlevel 1 (
            echo Error: Failed to install pip in venv.
            pause
            exit /b 1
        )
    )

    REM Install uv directly into venv
    "%VENV_DIR%\Scripts\python.exe" -m pip install uv
    if errorlevel 1 (
        echo Error: Failed to install uv in venv.
        pause
        exit /b 1
    )

    REM Confirm uv.exe exists after install
    if not exist "%UV_EXE%" (
        echo Error: uv.exe not found at expected location: %UV_EXE%
        pause
        exit /b 1
    )
)

echo uv is available.

REM Install main repo dependencies via uv
echo Installing main dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" -e "%REPO_DIR%"
if errorlevel 1 (
    echo Error: Failed to install main dependencies.
    pause
    exit /b 1
)

REM Install additional dependencies via uv
echo Installing additional dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" soundfile ffmpeg-python huggingface_hub ruaccent
if errorlevel 1 (
    echo Error: Failed to install additional dependencies.
    pause
    exit /b 1
)

REM Install PyTorch with CUDA support via uv
echo.
echo Reinstalling PyTorch with NVIDIA CUDA support...
uv pip uninstall torch torchvision torchaudio
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo Error: Failed to install PyTorch with CUDA support.
    pause
    exit /b 1
)
echo PyTorch for CUDA installed successfully.

REM Install bitsandbytes for 4-bit quantization
echo.
echo Installing bitsandbytes for int4 quantization support...
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" bitsandbytes

if errorlevel 1 (
    echo Error: Failed to install bitsandbytes.
    pause
    exit /b 1
)
echo bitsandbytes installed successfully.
echo Done.

REM Step 9: Apply modifications
echo.
echo [9/10] Applying custom Gradio demo file...
if exist "%~dp0gradio_demo.py" (
    copy /Y "%~dp0gradio_demo.py" "%REPO_DIR%\demo\gradio_demo.py" > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy 'gradio_demo.py'. Check file permissions.
        pause
        exit /b 1
    )
    echo Custom demo file applied successfully.
) else (
    echo Warning: 'gradio_demo.py' not found next to the installer. Using original file from repository.
)
echo Done.

REM Step 10: Copy voice samples from voice-pack
echo.
echo [10/10] Copying voice samples...
if exist "%~dp0voice-pack\" (
    echo Found 'voice-pack' directory. Copying contents...
    xcopy "%~dp0voice-pack" "%REPO_DIR%\demo\voices\" /E /I /Y > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy voice samples.
        pause
    ) else (
        echo Voice samples copied successfully.
        if "!CLEAN_VOICEPACK!"=="1" (
            echo Cleaning up 'voice-pack' directory...
            rmdir /s /q "%~dp0voice-pack"
            if exist "%~dp0voice-pack\" (
                echo Warning: Failed to delete 'voice-pack' directory.
            ) else (
                echo 'voice-pack' directory deleted.
            )
        )
    )
) else (
    echo Warning: 'voice-pack' directory not found. Skipping voice samples copy.
)

if exist "%~dp0gradio_demo.py" (
    del /f /q "%~dp0gradio_demo.py"
    if exist "%~dp0gradio_demo.py" (
        echo Warning: Failed to delete 'gradio_demo.py'.
    ) else (
        echo 'gradio_demo.py' deleted.
    )
) else (
    echo 'gradio_demo.py' not found. Skipping.
)

echo Done.

rmdir /s /q "%TEMP%" 2>nul
pause
goto main_menu

:install_vibe_cu121flashattn
cls
REM Step 1: Clean old installation if exists
echo [1/10] Preparing for installation: cleaning up old files...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%REPO_DIR%" rmdir /s /q "%REPO_DIR%"
if exist "%PYTHONUSERBASE%" rmdir /s /q "%PYTHONUSERBASE%"
if exist "%TEMP%" rmdir /s /q "%TEMP%"
if exist "%GIT_DIR%" rmdir /s /q "%GIT_DIR%"
if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
if "%CLEAN_PIP_CACHE%"=="1" (
    if exist "%PIP_CACHE_DIR%" rmdir /s /q "%PIP_CACHE_DIR%"
)

if "%CLEAN_HF_HOME%"=="1" (
    if exist "%HF_HOME%" rmdir /s /q "%HF_HOME%"
)

if "%CLEAN_VOICEPACK%"=="1" (
    if exist "%VOICEPACK_DIR%" rmdir /s /q "%VOICEPACK_DIR%"
)

if "%CLEAN_FFMPEG%"=="1" (
    if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%"
)
mkdir "%TEMP%"
echo Done.

set "EXPECTED_HASH=1FC64CA91B9B475AB0ADA72C9F7B3ADDBE69A6C8F520BE31425CF21841CCA369"
set "TMP_FILE=%INSTALL_DIR%PortableGit.tmp"
set "FINAL_FILE=%INSTALL_DIR%PortableGit.exe"

if not exist "%GIT_DIR%" (
    :: Проверка уже скачанного файла
    if exist "%FINAL_FILE%" (
        call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
        if "%HASH_MATCH%"=="1" (
            echo Git installer already downloaded and verified.
            goto :InstallGit
        ) else (
            echo Hash mismatch. Removing corrupted file.
            del "%FINAL_FILE%"
        )
    )

    :: Докачка
    if exist "%TMP_FILE%" (
        echo Resuming Git download...
    ) else (
        echo Starting Git download...
    )

    curl -C - --retry %MAX_RETRIES% --retry-delay 2 -L "%GIT_URL%" -o "%TMP_FILE%"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Git after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    move /Y "%TMP_FILE%" "%FINAL_FILE%"

    call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
    if "%HASH_MATCH%"=="0" (
        echo Error: Hash mismatch after download.
        del "%FINAL_FILE%"
        del "%TMP_FILE%"
        pause
        exit /b 1
    )

    :InstallGit
    "%FINAL_FILE%" -y -gm2 -nr -o"%GIT_DIR%"
    del "%FINAL_FILE%"
)

set "PATH=%GIT_DIR%\bin;%~dp0python_embedded;%~dp0ffmpeg\bin;%PATH%"
echo Done.

REM Step 3: Download and extract portable Python
echo.
echo [3/10] Downloading and setting up portable Python 3.11...
if not exist "%PYTHON_DIR%" (
    mkdir "%PYTHON_DIR%"
    call :DownloadFile "%PYTHON_URL%" "%INSTALL_DIR%python.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Python after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%python.zip' -DestinationPath '%PYTHON_DIR%'"
    del "%INSTALL_DIR%python.zip"

    echo.
    echo Configuring Python to use pip...
    powershell -Command "(Get-Content '%PYTHON_DIR%\python311._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python311._pth'"
    
    echo.
    echo Installing pip...
    call :DownloadFile "%GET_PIP_URL%" "%PYTHON_DIR%\get-pip.py"
     if %errorlevel% neq 0 (
        echo Error: Failed to download get-pip.py after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
)
echo Done.

REM Step 4: Download and extract ffmpeg
echo.
echo [4/10] Downloading and unpack ffmpeg binaries...
if not exist "%FFMPEG_DIR%" (
    mkdir "%FFMPEG_DIR%"
    call :DownloadFile "%FFMPEG_URL%" "%INSTALL_DIR%ffmpeg.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download FFMPEG after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%'"
    del "%INSTALL_DIR%ffmpeg.zip"

    echo.
)
echo Done.

REM Step 5: Download and extract VoicePack
echo.
echo [5/10] Downloading and unpack VoicePack...
if not exist "%VOICEPACK_DIR%" (
    mkdir "%VOICEPACK_DIR%"
    call :DownloadFile "%VOICEPACK_URL%" "%INSTALL_DIR%voice-pack.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download VoicePack after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%voice-pack.zip' -DestinationPath '%VOICEPACK_DIR%'"
    del "%INSTALL_DIR%voice-pack.zip"

    echo.
)
echo Done.

REM Step 6: Download and extract GradioDemo
echo.
echo [6/10] Downloading and unpack GradioDemo...
    call :DownloadFile "%GRADIODEMO_URL%" "%INSTALL_DIR%gradio_demo.py"
    if %errorlevel% neq 0 (
        echo Error: Failed to download GradioDemo after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    echo.
echo Done.

REM Step 7: Install virtualenv
echo.
echo [7/10] Installing virtualenv...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --no-cache-dir virtualenv
if %errorlevel% neq 0 (
    echo Error: Failed to install virtualenv after %MAX_RETRIES% attempts.
    pause
    exit /b 1
)
echo Done.

REM Step 8: Create virtual environment and install dependencies
echo.
echo [8/10] Cloning repository and installing dependencies...
if not exist "%REPO_DIR%" (
    call :RunCommandWithRetry "%GIT_DIR%\bin\git.exe" clone --depth 1 "%REPO_URL%" "%REPO_DIR%"
    if %errorlevel% neq 0 (
        echo Error: Failed to clone the VibeVoice repository after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
)
"%PYTHON_DIR%\python.exe" -m virtualenv --no-download "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo Error: Failed to create the virtual environment.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate"

echo Upgrading pip...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --cache-dir "%PIP_CACHE_DIR%" --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

echo Checking for uv in virtual environment...
set "UV_EXE=%VENV_DIR%\Scripts\uv.exe"

if not exist "%UV_EXE%" (
    echo 'uv' not found in venv. Attempting installation...

    REM Ensure pip is available in venv
    "%VENV_DIR%\Scripts\python.exe" -m ensurepip --default-pip >nul 2>&1
    if errorlevel 1 (
        echo Failed to bootstrap pip in venv. Trying get-pip.py...
        curl -L -o "%TEMP%\get-pip.py" "%GET_PIP_URL%"
        "%VENV_DIR%\Scripts\python.exe" "%TEMP%\get-pip.py"
        if errorlevel 1 (
            echo Error: Failed to install pip in venv.
            pause
            exit /b 1
        )
    )

    REM Install uv directly into venv
    "%VENV_DIR%\Scripts\python.exe" -m pip install uv
    if errorlevel 1 (
        echo Error: Failed to install uv in venv.
        pause
        exit /b 1
    )

    REM Confirm uv.exe exists after install
    if not exist "%UV_EXE%" (
        echo Error: uv.exe not found at expected location: %UV_EXE%
        pause
        exit /b 1
    )
)

echo uv is available.

REM Install main repo dependencies via uv
echo Installing main dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" -e "%REPO_DIR%"
if errorlevel 1 (
    echo Error: Failed to install main dependencies.
    pause
    exit /b 1
)

REM Install additional dependencies via uv
echo Installing additional dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" soundfile ffmpeg-python huggingface_hub ruaccent
if errorlevel 1 (
    echo Error: Failed to install additional dependencies.
    pause
    exit /b 1
)

REM Install PyTorch with CUDA support via uv
echo.
echo Reinstalling PyTorch with NVIDIA CUDA support...
uv pip uninstall torch torchvision torchaudio
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo Error: Failed to install PyTorch with CUDA support.
    pause
    exit /b 1
)
echo PyTorch for CUDA installed successfully.

REM Check einops
call "%~dp0venv\Scripts\python.exe" -m pip show einops >nul 2>&1
if errorlevel 1 (
    echo Installing einops...
    call :RunCommandWithRetry uv pip install einops --no-cache-dir
)

REM Check transformers
call "%~dp0venv\Scripts\python.exe" -m pip show transformers >nul 2>&1
if errorlevel 1 (
    echo Installing transformers...
    call :RunCommandWithRetry uv pip install transformers --no-cache-dir
)

REM Install FlashAttention 2.5.9 for torch 2.5.1 + CUDA 12.4.1 via uv
echo.
echo Installing FlashAttention 2.5.9 from prebuilt wheel via uv...

set "FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.9/flash_attn-2.5.9+cu124torch2.5-cp311-cp311-win_amd64.whl"

call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" ^
  "%FLASH_ATTN_URL%" --force-reinstall --no-deps

if errorlevel 1 (
    echo Error: Failed to install FlashAttention 2.5.9 via uv.
    pause
    exit /b 1
)
echo FlashAttention 2.5.9 installed successfully via uv.

call :RunCommandWithRetry uv pip install triton-windows --no-cache-dir

REM Install bitsandbytes for 4-bit quantization
echo.
echo Installing bitsandbytes for int4 quantization support...
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" bitsandbytes

if errorlevel 1 (
    echo Error: Failed to install bitsandbytes.
    pause
    exit /b 1
)
echo bitsandbytes installed successfully.
echo Done.

REM Step 9: Apply modifications
echo.
echo [9/10] Applying custom Gradio demo file...
if exist "%~dp0gradio_demo.py" (
    copy /Y "%~dp0gradio_demo.py" "%REPO_DIR%\demo\gradio_demo.py" > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy 'gradio_demo.py'. Check file permissions.
        pause
        exit /b 1
    )
    echo Custom demo file applied successfully.
) else (
    echo Warning: 'gradio_demo.py' not found next to the installer. Using original file from repository.
)
echo Done.

REM Step 10: Copy voice samples from voice-pack
echo.
echo [10/10] Copying voice samples...
if exist "%~dp0voice-pack\" (
    echo Found 'voice-pack' directory. Copying contents...
    xcopy "%~dp0voice-pack" "%REPO_DIR%\demo\voices\" /E /I /Y > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy voice samples.
        pause
    ) else (
        echo Voice samples copied successfully.
        if "!CLEAN_VOICEPACK!"=="1" (
            echo Cleaning up 'voice-pack' directory...
            rmdir /s /q "%~dp0voice-pack"
            if exist "%~dp0voice-pack\" (
                echo Warning: Failed to delete 'voice-pack' directory.
            ) else (
                echo 'voice-pack' directory deleted.
            )
        )
    )
) else (
    echo Warning: 'voice-pack' directory not found. Skipping voice samples copy.
)

if exist "%~dp0gradio_demo.py" (
    del /f /q "%~dp0gradio_demo.py"
    if exist "%~dp0gradio_demo.py" (
        echo Warning: Failed to delete 'gradio_demo.py'.
    ) else (
        echo 'gradio_demo.py' deleted.
    )
) else (
    echo 'gradio_demo.py' not found. Skipping.
)

echo Done.

rmdir /s /q "%TEMP%" 2>nul
pause
goto main_menu

:install_vibe_271cu128
cls
REM Step 1: Clean old installation if exists
echo [1/10] Preparing for installation: cleaning up old files...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%REPO_DIR%" rmdir /s /q "%REPO_DIR%"
if exist "%PYTHONUSERBASE%" rmdir /s /q "%PYTHONUSERBASE%"
if exist "%TEMP%" rmdir /s /q "%TEMP%"
if exist "%GIT_DIR%" rmdir /s /q "%GIT_DIR%"
if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
if "%CLEAN_PIP_CACHE%"=="1" (
    if exist "%PIP_CACHE_DIR%" rmdir /s /q "%PIP_CACHE_DIR%"
)

if "%CLEAN_HF_HOME%"=="1" (
    if exist "%HF_HOME%" rmdir /s /q "%HF_HOME%"
)

if "%CLEAN_VOICEPACK%"=="1" (
    if exist "%VOICEPACK_DIR%" rmdir /s /q "%VOICEPACK_DIR%"
)

if "%CLEAN_FFMPEG%"=="1" (
    if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%"
)
mkdir "%TEMP%"
echo Done.

set "EXPECTED_HASH=1FC64CA91B9B475AB0ADA72C9F7B3ADDBE69A6C8F520BE31425CF21841CCA369"
set "TMP_FILE=%INSTALL_DIR%PortableGit.tmp"
set "FINAL_FILE=%INSTALL_DIR%PortableGit.exe"

if not exist "%GIT_DIR%" (
    :: Проверка уже скачанного файла
    if exist "%FINAL_FILE%" (
        call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
        if "%HASH_MATCH%"=="1" (
            echo Git installer already downloaded and verified.
            goto :InstallGit
        ) else (
            echo Hash mismatch. Removing corrupted file.
            del "%FINAL_FILE%"
        )
    )

    :: Докачка
    if exist "%TMP_FILE%" (
        echo Resuming Git download...
    ) else (
        echo Starting Git download...
    )

    curl -C - --retry %MAX_RETRIES% --retry-delay 2 -L "%GIT_URL%" -o "%TMP_FILE%"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Git after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    move /Y "%TMP_FILE%" "%FINAL_FILE%"

    call :VerifyHash "%FINAL_FILE%" "%EXPECTED_HASH%"
    if "%HASH_MATCH%"=="0" (
        echo Error: Hash mismatch after download.
        del "%FINAL_FILE%"
        del "%TMP_FILE%"
        pause
        exit /b 1
    )

    :InstallGit
    "%FINAL_FILE%" -y -gm2 -nr -o"%GIT_DIR%"
    del "%FINAL_FILE%"
)

set "PATH=%GIT_DIR%\bin;%~dp0python_embedded;%~dp0ffmpeg\bin;%PATH%"
echo Done.

REM Step 3: Download and extract portable Python
echo.
echo [3/10] Downloading and setting up portable Python 3.11...
if not exist "%PYTHON_DIR%" (
    mkdir "%PYTHON_DIR%"
    call :DownloadFile "%PYTHON_URL%" "%INSTALL_DIR%python.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download Python after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%python.zip' -DestinationPath '%PYTHON_DIR%'"
    del "%INSTALL_DIR%python.zip"

    echo.
    echo Configuring Python to use pip...
    powershell -Command "(Get-Content '%PYTHON_DIR%\python311._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python311._pth'"
    
    echo.
    echo Installing pip...
    call :DownloadFile "%GET_PIP_URL%" "%PYTHON_DIR%\get-pip.py"
     if %errorlevel% neq 0 (
        echo Error: Failed to download get-pip.py after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
)
echo Done.

REM Step 4: Download and extract ffmpeg
echo.
echo [4/10] Downloading and unpack ffmpeg binaries...
if not exist "%FFMPEG_DIR%" (
    mkdir "%FFMPEG_DIR%"
    call :DownloadFile "%FFMPEG_URL%" "%INSTALL_DIR%ffmpeg.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download FFMPEG after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%'"
    del "%INSTALL_DIR%ffmpeg.zip"

    echo.
)
echo Done.

REM Step 5: Download and extract VoicePack
echo.
echo [5/10] Downloading and unpack VoicePack...
if not exist "%VOICEPACK_DIR%" (
    mkdir "%VOICEPACK_DIR%"
    call :DownloadFile "%VOICEPACK_URL%" "%INSTALL_DIR%voice-pack.zip"
    if %errorlevel% neq 0 (
        echo Error: Failed to download VoicePack after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%voice-pack.zip' -DestinationPath '%VOICEPACK_DIR%'"
    del "%INSTALL_DIR%voice-pack.zip"

    echo.
)
echo Done.

REM Step 6: Download and extract GradioDemo
echo.
echo [6/10] Downloading and unpack GradioDemo...
    call :DownloadFile "%GRADIODEMO_URL%" "%INSTALL_DIR%gradio_demo.py"
    if %errorlevel% neq 0 (
        echo Error: Failed to download GradioDemo after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )

    echo.
echo Done.

REM Step 7: Install virtualenv
echo.
echo [7/10] Installing virtualenv...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --no-cache-dir virtualenv
if %errorlevel% neq 0 (
    echo Error: Failed to install virtualenv after %MAX_RETRIES% attempts.
    pause
    exit /b 1
)
echo Done.

REM Step 8: Create virtual environment and install dependencies
echo.
echo [8/10] Cloning repository and installing dependencies...
if not exist "%REPO_DIR%" (
    call :RunCommandWithRetry "%GIT_DIR%\bin\git.exe" clone --depth 1 "%REPO_URL%" "%REPO_DIR%"
    if %errorlevel% neq 0 (
        echo Error: Failed to clone the VibeVoice repository after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
)
"%PYTHON_DIR%\python.exe" -m virtualenv --no-download "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo Error: Failed to create the virtual environment.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate"

echo Upgrading pip...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --cache-dir "%PIP_CACHE_DIR%" --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

echo Checking for uv in virtual environment...
set "UV_EXE=%VENV_DIR%\Scripts\uv.exe"

if not exist "%UV_EXE%" (
    echo 'uv' not found in venv. Attempting installation...

    REM Ensure pip is available in venv
    "%VENV_DIR%\Scripts\python.exe" -m ensurepip --default-pip >nul 2>&1
    if errorlevel 1 (
        echo Failed to bootstrap pip in venv. Trying get-pip.py...
        curl -L -o "%TEMP%\get-pip.py" "%GET_PIP_URL%"
        "%VENV_DIR%\Scripts\python.exe" "%TEMP%\get-pip.py"
        if errorlevel 1 (
            echo Error: Failed to install pip in venv.
            pause
            exit /b 1
        )
    )

    REM Install uv directly into venv
    "%VENV_DIR%\Scripts\python.exe" -m pip install uv
    if errorlevel 1 (
        echo Error: Failed to install uv in venv.
        pause
        exit /b 1
    )

    REM Confirm uv.exe exists after install
    if not exist "%UV_EXE%" (
        echo Error: uv.exe not found at expected location: %UV_EXE%
        pause
        exit /b 1
    )
)

echo uv is available.

REM Install main repo dependencies via uv
echo Installing main dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" -e "%REPO_DIR%"
if errorlevel 1 (
    echo Error: Failed to install main dependencies.
    pause
    exit /b 1
)

REM Install additional dependencies via uv
echo Installing additional dependencies.
call :RunCommandWithRetry "%UV_EXE%" pip install --cache-dir "%PIP_CACHE_DIR%" soundfile ffmpeg-python huggingface_hub ruaccent
if errorlevel 1 (
    echo Error: Failed to install additional dependencies.
    pause
    exit /b 1
)

REM Install PyTorch with CUDA support via uv
echo.
echo Reinstalling PyTorch with NVIDIA CUDA support...
uv pip uninstall torch torchvision torchaudio
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

if errorlevel 1 (
    echo Error: Failed to install PyTorch with CUDA support.
    pause
    exit /b 1
)
echo PyTorch for CUDA installed successfully.

REM Install bitsandbytes for 4-bit quantization
echo.
echo Installing bitsandbytes for int4 quantization support...
call :RunCommandWithRetry uv pip install --cache-dir "%PIP_CACHE_DIR%" bitsandbytes

if errorlevel 1 (
    echo Error: Failed to install bitsandbytes.
    pause
    exit /b 1
)
echo bitsandbytes installed successfully.
echo Done.

REM Step 9: Apply modifications
echo.
echo [9/10] Applying custom Gradio demo file...
if exist "%~dp0gradio_demo.py" (
    copy /Y "%~dp0gradio_demo.py" "%REPO_DIR%\demo\gradio_demo.py" > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy 'gradio_demo.py'. Check file permissions.
        pause
        exit /b 1
    )
    echo Custom demo file applied successfully.
) else (
    echo Warning: 'gradio_demo.py' not found next to the installer. Using original file from repository.
)
echo Done.

REM Step 10: Copy voice samples from voice-pack
echo.
echo [10/10] Copying voice samples...
if exist "%~dp0voice-pack\" (
    echo Found 'voice-pack' directory. Copying contents...
    xcopy "%~dp0voice-pack" "%REPO_DIR%\demo\voices\" /E /I /Y > nul
    if !errorlevel! neq 0 (
        echo Error: Failed to copy voice samples.
        pause
    ) else (
        echo Voice samples copied successfully.
        if "!CLEAN_VOICEPACK!"=="1" (
            echo Cleaning up 'voice-pack' directory...
            rmdir /s /q "%~dp0voice-pack"
            if exist "%~dp0voice-pack\" (
                echo Warning: Failed to delete 'voice-pack' directory.
            ) else (
                echo 'voice-pack' directory deleted.
            )
        )
    )
) else (
    echo Warning: 'voice-pack' directory not found. Skipping voice samples copy.
)

if exist "%~dp0gradio_demo.py" (
    del /f /q "%~dp0gradio_demo.py"
    if exist "%~dp0gradio_demo.py" (
        echo Warning: Failed to delete 'gradio_demo.py'.
    ) else (
        echo 'gradio_demo.py' deleted.
    )
) else (
    echo 'gradio_demo.py' not found. Skipping.
)

echo Done.

rmdir /s /q "%TEMP%" 2>nul
pause
goto main_menu

:start_vibe
cls
chcp 65001 > nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PATH=%~dp0PortableGit\bin;%~dp0python_embedded;%~dp0ffmpeg\bin;%PATH%"
set "TEMP=%~dp0temp"
set "TMP=%~dp0temp"
set "GRADIO_TEMP_DIR=%~dp0temp"
set "HF_HOME=%~dp0huggingface_cache"
set "PIP_CACHE_DIR=%~dp0cache"
set "PYTHONUSERBASE=%~dp0python_user_base"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "models" mkdir "models"
where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ffprobe not found in PATH. Audio features may fail.
    pause
)
call "venv\Scripts\activate.bat"
cd VibeVoice_repo
echo.
echo Starting VibeVoice...
echo.
python demo\gradio_demo.py --device cuda
pause

:start_vibe_low
cls
chcp 65001 > nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PATH=%~dp0PortableGit\bin;%~dp0python_embedded;%~dp0ffmpeg\bin;%PATH%"
set "TEMP=%~dp0temp"
set "TMP=%~dp0temp"
set "GRADIO_TEMP_DIR=%~dp0temp"
set "HF_HOME=%~dp0huggingface_cache"
set "PIP_CACHE_DIR=%~dp0cache"
set "PYTHONUSERBASE=%~dp0python_user_base"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "models" mkdir "models"
where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ffprobe not found in PATH. Audio features may fail.
    pause
)
call "venv\Scripts\activate.bat"
cd VibeVoice_repo
echo.
echo Starting VibeVoice...
echo.
python demo\gradio_demo.py --device cpu --inference_steps 20
pause

:DownloadFile
set "URL=%~1"
set "OutputFile=%~2"
for /L %%i in (1,1,%MAX_RETRIES%) do (
    echo Attempt %%i of %MAX_RETRIES% to download %OutputFile%...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('!URL!', '!OutputFile!')"
    if !errorlevel! equ 0 (
        echo Download successful.
        exit /b 0
    )
    echo Download failed. Retrying in %RETRY_DELAY% seconds...
    timeout /t %RETRY_DELAY% /nobreak > nul
)
exit /b 1

:RunCommandWithRetry
set "command_to_run=%*"
for /L %%i in (1,1,%MAX_RETRIES%) do (
    echo Attempt %%i of %MAX_RETRIES%: Running '%command_to_run%'...
    %command_to_run%
    if !errorlevel! equ 0 (
        echo Command successful.
        exit /b 0
    )
    echo Command failed. Retrying in %RETRY_DELAY% seconds...
    timeout /t %RETRY_DELAY% /nobreak > nul
)
exit /b 1

:VerifyHash
:: %~1 — путь к файлу
:: %~2 — ожидаемый хеш
setlocal EnableDelayedExpansion
set "FILE=%~1"
set "EXPECTED=%~2"
set "HASH="

:: Попытка через certutil
for /f "tokens=1" %%A in ('certutil -hashfile "%FILE%" SHA256 ^| find /i /v "SHA256"') do (
    set "HASH=%%A"
    goto :CompareHash
)

:: Fallback через PowerShell
for /f %%A in ('powershell -command "(Get-FileHash -Algorithm SHA256 -Path '%FILE%').Hash"') do (
    set "HASH=%%A"
)

:CompareHash
echo Computed hash: [!HASH!]
echo Expected hash: [!EXPECTED!]
if /i "!HASH!"=="!EXPECTED!" (
    endlocal & set "HASH_MATCH=1"
) else (
    endlocal & set "HASH_MATCH=0"
)
goto :eof