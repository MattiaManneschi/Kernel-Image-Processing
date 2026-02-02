@echo off
REM =============================================================================
REM Run Advanced Tests - Windows
REM With automatic dependency installation
REM =============================================================================

setlocal EnableDelayedExpansion

REM Change to script directory
cd /d "%~dp0"

echo ==============================================
echo   Kernel Image Processing - Advanced Tests
echo ==============================================
echo.

REM =============================================================================
REM Check dependencies
REM =============================================================================

echo Checking dependencies...
echo.

set MISSING_DEPS=

REM Check for CUDA
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] CUDA toolkit not found
    set MISSING_DEPS=!MISSING_DEPS! CUDA
) else (
    echo [OK] CUDA toolkit found
)

REM Check for Visual Studio / cl.exe
where cl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Visual Studio C++ compiler not found
    set MISSING_DEPS=!MISSING_DEPS! VisualStudio
) else (
    echo [OK] Visual Studio C++ compiler found
)

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Python not found
        set MISSING_DEPS=!MISSING_DEPS! Python
    ) else (
        echo [OK] Python3 found
    )
) else (
    echo [OK] Python found
)

REM Check for curl
where curl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] curl not found
    set MISSING_DEPS=!MISSING_DEPS! curl
) else (
    echo [OK] curl found
)

echo.

REM =============================================================================
REM Show installation instructions if dependencies missing
REM =============================================================================

if defined MISSING_DEPS (
    echo ==============================================
    echo   Missing Dependencies: !MISSING_DEPS!
    echo ==============================================
    echo.

    echo !MISSING_DEPS! | findstr /C:"CUDA" >nul && (
        echo CUDA Toolkit:
        echo   Download from: https://developer.nvidia.com/cuda-downloads
        echo   Select Windows ^> x86_64 ^> Your Windows version ^> exe [local]
        echo.
    )

    echo !MISSING_DEPS! | findstr /C:"VisualStudio" >nul && (
        echo Visual Studio with C++:
        echo   Download from: https://visualstudio.microsoft.com/downloads/
        echo   Select "Desktop development with C++"
        echo   OR use WSL2 with Ubuntu for easier setup
        echo.
    )

    echo !MISSING_DEPS! | findstr /C:"Python" >nul && (
        echo Python:
        echo   Download from: https://www.python.org/downloads/
        echo   Make sure to check "Add Python to PATH" during install
        echo.

        REM Try to install with winget
        where winget >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo Attempting to install Python with winget...
            winget install Python.Python.3.11
        )
    )

    echo !MISSING_DEPS! | findstr /C:"curl" >nul && (
        echo curl:
        echo   Usually included in Windows 10+
        echo   Or download from: https://curl.se/windows/
        echo.
    )

    echo.
    echo Please install missing dependencies and run this script again.
    echo.
    echo TIP: For easier setup, consider using WSL2 ^(Windows Subsystem for Linux^)
    echo      and run the Linux version of this script.
    echo.
    pause
    exit /b 1
)

REM =============================================================================
REM Install Python packages
REM =============================================================================

echo Checking Python packages...

python -c "import matplotlib" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing matplotlib...
    pip install matplotlib
)

python -c "import pandas" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing pandas...
    pip install pandas
)

python -c "import numpy" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing numpy...
    pip install numpy
)

echo [OK] Python packages ready
echo.

REM =============================================================================
REM Download Kodak images if missing
REM =============================================================================

set KODAK_DIR=images\input\kodak

if not exist "%KODAK_DIR%" mkdir "%KODAK_DIR%"

REM Count existing images
set /a IMG_COUNT=0
for %%f in (%KODAK_DIR%\kodim*.png) do set /a IMG_COUNT+=1

if %IMG_COUNT% LSS 20 (
    echo Downloading Kodak test images...

    for /L %%i in (1,1,24) do (
        set NUM=0%%i
        set NUM=!NUM:~-2!

        if not exist "%KODAK_DIR%\kodim!NUM!.png" (
            echo   Downloading kodim!NUM!.png...
            curl -s -f -o "%KODAK_DIR%\kodim!NUM!.png" "http://r0k.us/graphics/kodak/kodak/kodim!NUM!.png" 2>nul
            if !ERRORLEVEL! NEQ 0 (
                echo     Failed to download kodim!NUM!.png
            )
        )
    )

    echo.
    echo Kodak images downloaded.
    echo.
) else (
    echo [OK] Kodak images found in %KODAK_DIR%
    echo.
)

REM =============================================================================
REM Check if executable exists
REM =============================================================================

if not exist "bin\imgproc.exe" (
    echo.
    echo ==============================================
    echo   Executable not found: bin\imgproc.exe
    echo ==============================================
    echo.
    echo Please build the project first:
    echo.
    echo Option 1 - Visual Studio:
    echo   1. Open the project in Visual Studio
    echo   2. Build ^> Build Solution
    echo.
    echo Option 2 - WSL2 ^(Recommended^):
    echo   1. Open WSL terminal in this folder
    echo   2. Run: make clean ^&^& make all
    echo   3. Run: ./run_advanced_tests.sh
    echo.
    pause
    exit /b 1
)

echo [OK] Executable found: bin\imgproc.exe
echo.

REM =============================================================================
REM Run the tests
REM =============================================================================

echo ==============================================
echo   Running Advanced Tests
echo ==============================================
echo.

REM Check if Kodak images exist
set /a IMG_COUNT=0
for %%f in (%KODAK_DIR%\kodim*.png) do set /a IMG_COUNT+=1

if %IMG_COUNT% GEQ 1 (
    echo Using Kodak images from %KODAK_DIR%
    echo.
    bin\imgproc.exe --advanced-tests --images-dir images/input/kodak
) else (
    echo WARNING: Running with synthetic images
    echo.
    bin\imgproc.exe --advanced-tests
)

echo.
echo ==============================================
echo   Generating Filtered Images (All Kernels)
echo ==============================================
echo.

REM Pick a RANDOM Kodak image
set /a COUNT=0
for %%f in (%KODAK_DIR%\kodim*.png) do set /a COUNT+=1

if %COUNT% GEQ 1 (
    REM Generate random index
    set /a RAND_IDX=%RANDOM% %% %COUNT%

    REM Find the image at that index
    set /a CURR_IDX=0
    for %%f in (%KODAK_DIR%\kodim*.png) do (
        if !CURR_IDX! EQU !RAND_IDX! (
            set INPUT_IMG=%%f
            goto :run_generate
        )
        set /a CURR_IDX+=1
    )
)

:run_generate
if defined INPUT_IMG (
    echo Randomly selected: %INPUT_IMG%
    bin\imgproc.exe -i "%INPUT_IMG%" --generate-all
) else (
    echo No Kodak images found, skipping image generation
)

echo.
echo ==============================================
echo   Tests Complete!
echo ==============================================
echo.
echo Results saved to:
echo   - results\advanced_tests\advanced_results.csv
echo   - images\output\^<image_name^>\
echo.

pause