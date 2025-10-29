# Build and Publish ChunkFlow to PyPI

Complete guide for building and publishing ChunkFlow to PyPI using **Windows CMD**.

## Prerequisites

### 1. Install Build Tools

```cmd
REM Upgrade pip
python -m pip install --upgrade pip

REM Install build tools
python -m pip install --upgrade build twine setuptools wheel
```

### 2. Create PyPI Account

1. Sign up at https://pypi.org/account/register/
2. Verify your email
3. Enable 2FA (recommended)

### 3. Create API Token

1. Go to https://pypi.org/manage/account/token/
2. Create new token for "chunk-flow" project (or account-wide token)
3. Copy token (starts with `pypi-`)
4. Save it securely - you'll use this instead of password

## Pre-Release Checklist

Before building and publishing, ensure:

```cmd
REM 1. All tests pass
pytest

REM 2. Code is formatted
black chunk_flow tests examples
isort chunk_flow tests examples

REM 3. Linting passes
ruff check chunk_flow tests examples
mypy chunk_flow

REM 4. Clean notebooks (remove execution outputs and kernel metadata)
python clean_notebooks.py

REM 5. Check version numbers match
REM   - chunk_flow/__init__.py
REM   - pyproject.toml
REM   - CHANGELOG.md
```

## Build Process

### Step 1: Clean Previous Builds

```cmd
REM Remove old build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist chunk_flow.egg-info rmdir /s /q chunk_flow.egg-info
if exist .eggs rmdir /s /q .eggs

REM Clean Python cache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
```

### Step 2: Build Distribution

```cmd
REM Build source distribution and wheel
python -m build

REM This creates:
REM   dist/chunk-flow-0.1.0.tar.gz (source distribution)
REM   dist/chunk_flow-0.1.0-py3-none-any.whl (wheel)
```

### Step 3: Check Distribution

```cmd
REM Check that the distribution is valid
twine check dist/*

REM Output should show:
REM   Checking dist/chunk-flow-0.1.0.tar.gz: PASSED
REM   Checking dist/chunk_flow-0.1.0-py3-none-any.whl: PASSED
```

## Publish to Test PyPI (Optional but Recommended)

Test your package on TestPyPI before uploading to the real PyPI.

### Step 1: Create TestPyPI Account

1. Sign up at https://test.pypi.org/account/register/
2. Create API token at https://test.pypi.org/manage/account/token/

### Step 2: Upload to TestPyPI

```cmd
REM Upload to TestPyPI
twine upload --repository testpypi dist/*

REM When prompted:
REM   Username: __token__
REM   Password: <paste your TestPyPI token>
```

### Step 3: Test Installation from TestPyPI

```cmd
REM Create test environment
python -m venv test_env
test_env\Scripts\activate

REM Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chunk-flow

REM Test it works
python -c "from chunk_flow.chunking import StrategyRegistry; print('✓ Import successful')"

REM Clean up
deactivate
rmdir /s /q test_env
```

## Publish to PyPI (Production)

### Step 1: Upload to PyPI

```cmd
REM Upload to production PyPI
twine upload dist/*

REM When prompted:
REM   Username: __token__
REM   Password: <paste your PyPI token>
```

**Alternatively, use a `.pypirc` file:**

Create `%USERPROFILE%\.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```

Then upload without prompts:

```cmd
twine upload dist/*
```

### Step 2: Verify on PyPI

1. Visit https://pypi.org/project/chunk-flow/
2. Check version number, description, links
3. Verify README renders correctly

### Step 3: Test Installation

```cmd
REM Create clean test environment
python -m venv verify_env
verify_env\Scripts\activate

REM Install from PyPI
pip install chunk-flow

REM Test basic functionality
python -c "from chunk_flow.chunking import StrategyRegistry; print('✓ ChunkFlow installed successfully')"

REM Test with extras
pip install chunk-flow[all]

REM Clean up
deactivate
rmdir /s /q verify_env
```

## Post-Release

### Step 1: Create Git Tag

```cmd
REM Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0"

REM Push tag to GitHub
git push origin v0.1.0
```

### Step 2: Create GitHub Release

1. Go to https://github.com/YOUR-USERNAME/chunk-flow/releases/new
2. Select tag: `v0.1.0`
3. Release title: `ChunkFlow v0.1.0`
4. Copy relevant section from CHANGELOG.md
5. Upload distribution files from `dist/` folder:
   - `chunk-flow-0.1.0.tar.gz`
   - `chunk_flow-0.1.0-py3-none-any.whl`
6. Click "Publish release"

### Step 3: Update Version for Development

Update version to next dev version:

**chunk_flow/__init__.py:**
```python
__version__ = "0.2.0.dev0"
```

**pyproject.toml:**
```toml
version = "0.2.0.dev0"
```

Commit:
```cmd
git add chunk_flow/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.2.0.dev0"
git push origin main
```

## Complete Build & Publish Script (Windows CMD)

Save as `publish.bat`:

```cmd
@echo off
echo ===================================
echo ChunkFlow Build and Publish Script
echo ===================================
echo.

REM Step 1: Clean
echo [1/6] Cleaning old builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist chunk_flow.egg-info rmdir /s /q chunk_flow.egg-info

REM Step 2: Clean notebooks
echo [2/6] Cleaning Jupyter notebooks...
python clean_notebooks.py

REM Step 3: Run tests
echo [3/6] Running tests...
pytest
if %errorlevel% neq 0 (
    echo ERROR: Tests failed!
    exit /b 1
)

REM Step 4: Build
echo [4/6] Building distribution...
python -m build
if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    exit /b 1
)

REM Step 5: Check
echo [5/6] Checking distribution...
twine check dist/*
if %errorlevel% neq 0 (
    echo ERROR: Distribution check failed!
    exit /b 1
)

REM Step 6: Upload
echo [6/6] Uploading to PyPI...
echo.
echo IMPORTANT: You'll need to enter your PyPI credentials
echo   Username: __token__
echo   Password: <your PyPI token>
echo.
pause

twine upload dist/*

if %errorlevel% neq 0 (
    echo ERROR: Upload failed!
    exit /b 1
)

echo.
echo ===================================
echo ✓ SUCCESS! Package published to PyPI
echo ===================================
echo.
echo Next steps:
echo 1. Create git tag: git tag -a v0.1.0 -m "Release v0.1.0"
echo 2. Push tag: git push origin v0.1.0
echo 3. Create GitHub release
echo.
```

Run with:
```cmd
publish.bat
```

## Troubleshooting

### Issue: "Invalid distribution"

**Solution:**
```cmd
REM Ensure pyproject.toml and __init__.py versions match
REM Clean and rebuild
rmdir /s /q build dist *.egg-info
python -m build
```

### Issue: "Package name already exists"

**Solution:**
- Package name `chunk-flow` might be taken
- Try alternative names or contact PyPI support

### Issue: "Credential error"

**Solution:**
```cmd
REM Make sure you're using token authentication
REM Username: __token__
REM Password: pypi-... (your actual token)
```

### Issue: "File already exists"

**Solution:**
- You cannot re-upload the same version
- Increment version in `__init__.py` and `pyproject.toml`
- Rebuild and re-upload

### Issue: "README not rendering"

**Solution:**
- Check README.md uses standard markdown
- Validate with: https://commonmark.org/help/
- Ensure `long_description_content_type = "text/markdown"` in pyproject.toml

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New features
- `0.9.0` → `1.0.0`: First stable release
- `1.0.0` → `2.0.0`: Breaking changes

## Resources

- **PyPI Package**: https://pypi.org/project/chunk-flow/
- **PyPI Help**: https://pypi.org/help/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Docs**: https://twine.readthedocs.io/

---

**Ready to publish? Follow the checklist and you're good to go!** 🚀
