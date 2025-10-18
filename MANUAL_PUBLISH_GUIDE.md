# Manual Publishing Guide - Windows CMD

**Step-by-step commands to publish ChunkFlow to PyPI**

---

## ⚠️ CRITICAL: Security First

Your PyPI token was exposed. You **MUST** fix this before publishing:

### Step 1: Revoke Old Token

1. Go to: https://pypi.org/manage/account/token/
2. Find the token named "chunk-flow" (or similar)
3. Click "Remove" / "Revoke"

### Step 2: Create New Token

1. Go to: https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `chunk-flow`
4. Scope: **Entire account** (or specify "chunk-flow" if package already exists)
5. Click "Add token"
6. **Copy the token immediately** (you'll only see it once)

### Step 3: Update .pypirc

Open Notepad and create/update `.pypirc`:

```cmd
notepad %USERPROFILE%\.pypirc
```

Paste this content (replace `YOUR-NEW-TOKEN-HERE` with your actual token):

```
[pypi]
username = __token__
password = pypi-YOUR-NEW-TOKEN-HERE
```

**Save and close** Notepad.

---

## 📝 Publishing Steps

### Step 1: Navigate to Project

```cmd
cd "C:\Users\Lenovo i7\Desktop\my_personal_projects\chunckerflow"
```

Verify you're in the right directory:

```cmd
dir
```

You should see: `pyproject.toml`, `chunk_flow/`, `README.md`, etc.

---

### Step 2: Update Changelog

Open the changelog:

```cmd
notepad CHANGELOG.md
```

Find line 8 (or near the top) and update the date:

**Change from:**
```markdown
## [0.1.0] - UNRELEASED
```

**Change to:**
```markdown
## [0.1.0] - 2025-01-18
```

(Use today's actual date)

**Save and close** Notepad.

---

### Step 3: Clean Notebooks (Optional)

If you have Jupyter notebooks in `examples/jupyter/`:

```cmd
python clean_notebooks.py
```

If you get an error, skip this step.

---

### Step 4: Clean Old Builds

Remove old build artifacts:

```cmd
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist chunk_flow.egg-info rmdir /s /q chunk_flow.egg-info
```

---

### Step 5: Install/Update Build Tools

Make sure you have the latest build tools:

```cmd
pip install --upgrade build twine
```

Wait for installation to complete.

---

### Step 6: Build Package

Build the distribution files:

```cmd
python -m build
```

**Expected output:**
```
* Creating venv isolated environment...
* Installing packages in isolated environment... (hatchling)
* Getting build dependencies for sdist...
* Building sdist...
* Building wheel from sdist
* Creating venv isolated environment...
* Installing packages in isolated environment... (hatchling)
* Getting build dependencies for wheel...
* Building wheel...
Successfully built chunk_flow-0.1.0.tar.gz and chunk_flow-0.1.0-py3-none-any.whl
```

Verify build files were created:

```cmd
dir dist
```

You should see:
- `chunk_flow-0.1.0-py3-none-any.whl`
- `chunk_flow-0.1.0.tar.gz`

---

### Step 7: Validate Package

Check that the package is valid:

```cmd
twine check dist/*
```

**Expected output:**
```
Checking dist\chunk_flow-0.1.0-py3-none-any.whl: PASSED
Checking dist\chunk_flow-0.1.0.tar.gz: PASSED
```

If you see any FAILED messages, **STOP** and fix the issues.

---

### Step 8: Commit Changes

Add all changes to git:

```cmd
git add .
```

Commit with a message:

```cmd
git commit -m "chore: prepare release v0.1.0"
```

Push to GitHub:

```cmd
git push origin main
```

---

### Step 9: Wait for GitHub Actions (IMPORTANT!)

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Find the workflow run triggered by your push
4. **Wait for all checks to pass** (green checkmarks ✅)
5. This usually takes 3-5 minutes

**Do NOT continue** until all tests pass!

---

### Step 10: Upload to PyPI

Upload to PyPI (this is it!):

```cmd
twine upload dist/*
```

**Expected prompts/output:**

If `.pypirc` is configured correctly:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading chunk_flow-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading chunk_flow-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://pypi.org/project/chunk-flow/0.1.0/
```

If you get prompted for username/password:
- Username: `__token__`
- Password: `pypi-YOUR-NEW-TOKEN-HERE` (paste your token)

**Common errors:**

- **"Invalid or non-existent authentication"**: Your `.pypirc` token is wrong. Go back to Step 3.
- **"File already exists"**: Version 0.1.0 already published. You need to bump the version in `pyproject.toml`.
- **"Package name already taken"**: Someone else owns `chunk-flow`. You need a different name.

---

### Step 11: Create Git Tag

Tag the release:

```cmd
git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release"
```

Push the tag to GitHub:

```cmd
git push origin v0.1.0
```

---

### Step 12: Verify Publication

Check that your package is live:

1. Open browser: https://pypi.org/project/chunk-flow/
2. You should see version 0.1.0
3. Try installing in a fresh environment:

```cmd
pip install chunk-flow
```

---

## 🎉 Success!

Your package is now published on PyPI!

**Package URL:** https://pypi.org/project/chunk-flow/

**Install command:**
```bash
pip install chunk-flow
```

---

## 📋 Quick Reference (All Commands)

For future releases, here's the complete command sequence:

```cmd
REM Navigate
cd "C:\Users\Lenovo i7\Desktop\my_personal_projects\chunckerflow"

REM Update CHANGELOG.md date manually
notepad CHANGELOG.md

REM Clean builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Upgrade tools
pip install --upgrade build twine

REM Build
python -m build

REM Validate
twine check dist/*

REM Commit
git add .
git commit -m "chore: prepare release v0.1.X"
git push origin main

REM Wait for GitHub Actions to pass!

REM Upload
twine upload dist/*

REM Tag
git tag -a v0.1.X -m "Release v0.1.X"
git push origin v0.1.X
```

---

## ⚠️ Before Next Release

For version 0.1.1, 0.2.0, etc.:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or 0.2.0
   ```

2. Add new section in `CHANGELOG.md`:
   ```markdown
   ## [0.1.1] - 2025-XX-XX

   ### Fixed
   - Bug fixes here

   ### Added
   - New features here
   ```

3. Follow all steps above

---

## 🆘 Troubleshooting

**Q: `python -m build` fails**
- A: Make sure you have Python 3.9+ installed: `python --version`
- A: Install build: `pip install build`

**Q: `twine` not found**
- A: Install it: `pip install twine`

**Q: Upload fails with authentication error**
- A: Check `.pypirc` file exists: `type %USERPROFILE%\.pypirc`
- A: Revoke and create a new PyPI token

**Q: "File already exists" error**
- A: You can't re-upload the same version. Bump version in `pyproject.toml`

**Q: GitHub Actions failing**
- A: Fix the failing tests before publishing
- A: Check the Actions tab on GitHub for error details

---

**Need help?** Check:
- Full guide: `docs/EASY_RELEASE_GUIDE.md`
- Security info: `docs/SECURITY_WARNING.md`
- Project structure: `FINAL_STRUCTURE.md`
