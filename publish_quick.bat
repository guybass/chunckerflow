@echo off
REM ===================================
REM ChunkFlow Quick Publish Script
REM (No local tests - relies on GitHub Actions)
REM ===================================

REM Change to the directory where this script is located
cd /d "%~dp0"

echo.
echo ===================================
echo ChunkFlow Quick Publish Script
echo ===================================
echo.
echo Working directory: %CD%
echo.
echo NOTE: This script skips local tests.
echo Tests will run automatically on GitHub Actions when you push to main.
echo.

REM Step 1: Clean old builds
echo [1/4] Cleaning old builds...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist chunk_flow.egg-info rmdir /s /q chunk_flow.egg-info 2>nul
if exist .eggs rmdir /s /q .eggs 2>nul
echo     Done.
echo.

REM Step 2: Clean notebooks
echo [2/4] Cleaning Jupyter notebooks...
if exist clean_notebooks.py (
    python clean_notebooks.py
    if %errorlevel% neq 0 (
        echo     WARNING: Notebook cleaning had issues, continuing...
    ) else (
        echo     Done.
    )
) else (
    echo     clean_notebooks.py not found - SKIPPING
)
echo.

REM Step 3: Build distribution
echo [3/4] Building distribution packages...
python -m build
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo     Build complete.
echo.

REM Step 4: Check distribution
echo [4/4] Validating distribution packages...
twine check dist/*
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Distribution validation failed!
    pause
    exit /b 1
)
echo     Validation passed.
echo.

REM Upload to PyPI
echo ===================================
echo Ready to upload to PyPI
echo ===================================
echo.
echo Files to upload:
dir /b dist\*
echo.
echo Checking for .pypirc configuration...
if exist "%USERPROFILE%\.pypirc" (
    echo   ✓ Found .pypirc - will use stored credentials
    echo   No password prompt needed!
) else (
    echo   ⚠ .pypirc not found
    echo.
    echo   You will be prompted for credentials:
    echo     Username: __token__
    echo     Password: pypi-... (paste your PyPI token)
    echo.
    echo   TIP: Create %USERPROFILE%\.pypirc to avoid prompts
    echo   See docs\SECURITY_WARNING.md for instructions
)
echo.
echo Press any key to continue with upload, or Ctrl+C to cancel...
pause >nul
echo.

twine upload dist/*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Upload to PyPI failed!
    echo.
    echo Common issues:
    echo   - Invalid token
    echo   - Version already exists (increment version number)
    echo   - Network connectivity issues
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================
echo ✓ SUCCESS! Package published to PyPI
echo ===================================
echo.
echo Package available at: https://pypi.org/project/chunk-flow/
echo.
echo Next steps:
echo   1. Create git tag: git tag -a v0.1.0 -m "Release v0.1.0"
echo   2. Push to GitHub: git push origin main --tags
echo   3. GitHub Actions will run tests automatically
echo   4. Create GitHub release with dist/ files
echo   5. Update version to next dev version (e.g., 0.2.0.dev0)
echo.
pause
