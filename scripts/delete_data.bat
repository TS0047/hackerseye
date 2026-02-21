@echo off
REM Delete the entire data\ folder

cd /d "%~dp0.."

echo ============================================================
echo Delete Data Folder
echo ============================================================
echo Target: %CD%\data
echo.

if not exist "data" (
    echo data\ folder does not exist. Nothing to delete.
    pause
    exit /b 0
)

echo WARNING: This will permanently delete the entire data\ folder.
echo.
set /p CONFIRM="Proceed? [y/N]: "

if /i not "%CONFIRM%"=="y" (
    if /i not "%CONFIRM%"=="yes" (
        echo Cancelled.
        pause
        exit /b 0
    )
)

echo.
echo Deleting data\ folder...
rmdir /s /q "data" 2>nul

if not exist "data" (
    echo [OK] Successfully deleted data\ folder
) else (
    echo [ERROR] Failed to delete data\ folder
)

echo.
pause
