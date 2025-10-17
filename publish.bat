@echo off
REM ===================================
REM ChunkFlow Build and Publish Script
REM ===================================

echo.
echo ===================================
echo ChunkFlow Build and Publish Script
echo ===================================
echo.

REM Step 1: Clean old builds
echo [1/6] Cleaning old builds...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist chunk_flow.egg-info rmdir /s /q chunk_flow.egg-info 2>nul
if exist .eggs rmdir /s /q .eggs 2>nul
echo     Done.
echo.

REM Step 2: Clean notebooks
echo [2/6] Cleaning Jupyter notebooks...
python clean_notebooks.py
if %errorlevel% neq 0 (
    echo     WARNING: Notebook cleaning had issues, continuing...
)
echo.

REM Step 3: Run tests
echo [3/6] Running tests...
pytest -q
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Tests failed! Fix tests before publishing.
    pause
    exit /b 1
)
echo     Tests passed.
echo.

REM Step 4: Build distribution
echo [4/6] Building distribution packages...
python -m build
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo     Build complete.
echo.

REM Step 5: Check distribution
echo [5/6] Validating distribution packages...
twine check dist/*
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Distribution validation failed!
    pause
    exit /b 1
)
echo     Validation passed.
echo.

REM Step 6: Upload to PyPI
echo [6/6] Uploading to PyPI...
echo.
echo IMPORTANT: You need your PyPI API token
echo.
echo   When prompted, enter:
echo     Username: __token__
echo     Password: pypi-... (paste your PyPI token)
echo.
echo Press any key to continue with upload...
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
echo   2. Push tag: git push origin v0.1.0
echo   3. Create GitHub release with dist/ files
echo   4. Update version to next dev version (e.g., 0.2.0.dev0)
echo.
pause
