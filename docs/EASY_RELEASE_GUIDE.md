# 🚀 Easy Release Guide - ChunkFlow to PyPI

**Follow these steps in order. Each step takes 2-5 minutes.**

---

## ⚠️ BEFORE YOU START

### 🔴 CRITICAL: Security Check

**Your PyPI token was exposed. You MUST do this first:**

1. Go to: https://pypi.org/manage/account/token/
2. Find and **DELETE** the token you shared earlier
3. Create a **NEW** token:
   - Token name: `chunk-flow`
   - Scope: **"Entire account"** (for first publish)
   - Click **"Add token"**
   - **COPY the token** (you only see it once!)

4. Update your `.pypirc` file:
   ```cmd
   notepad %USERPROFILE%\.pypirc
   ```

   Replace the password line:
   ```ini
   [pypi]
   username = __token__
   password = pypi-YOUR-NEW-TOKEN-HERE
   ```

   **Save and close.**

✅ **Token security fixed? Continue below.**

---

## 📋 Release Checklist

Copy this and check off as you go:

```
[ ] Step 1: Security - Token revoked and .pypirc updated
[ ] Step 2: Version - Updated version number
[ ] Step 3: Changelog - Updated CHANGELOG.md
[ ] Step 4: Commit - All changes committed
[ ] Step 5: Push - Pushed to GitHub
[ ] Step 6: Actions - GitHub Actions passed
[ ] Step 7: Publish - Published to PyPI
[ ] Step 8: Tag - Created git tag
[ ] Step 9: Release - Created GitHub release
[ ] Step 10: Celebrate - You're live on PyPI! 🎉
```

---

## 📝 Step-by-Step Release

### Step 1: Update Version Number (2 minutes)

**Edit these 2 files:**

**File 1:** `chunk_flow/__init__.py`
```python
__version__ = "0.1.0"  # Change this
```

**File 2:** `pyproject.toml`
```toml
version = "0.1.0"  # Change this (line 4)
```

**Make them match!** Both should say `0.1.0`

---

### Step 2: Update Changelog (3 minutes)

**Edit:** `CHANGELOG.md`

Change line 8 from:
```markdown
## [0.1.0] - 2024-01-XX
```

To today's date:
```markdown
## [0.1.0] - 2025-01-17
```

**That's it!** The changelog is already complete.

---

### Step 3: Commit Your Changes (2 minutes)

**Open Command Prompt in project folder:**

```cmd
cd "C:\Users\Lenovo i7\Desktop\my_personal_projects\chunckerflow"

git add .
git commit -m "chore: prepare release v0.1.0"
```

**Expected output:**
```
[main abc1234] chore: prepare release v0.1.0
 2 files changed, 2 insertions(+), 2 deletions(-)
```

---

### Step 4: Push to GitHub (1 minute)

```cmd
git push origin main
```

**Expected output:**
```
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
...
To https://github.com/YOUR-USERNAME/chunk-flow.git
   abc1234..def5678  main -> main
```

---

### Step 5: Wait for GitHub Actions (3-5 minutes)

**Go to:** https://github.com/YOUR-USERNAME/chunk-flow/actions

**Wait for the green checkmark ✅**

You'll see tests running on:
- Python 3.9, 3.10, 3.11, 3.12
- Linux, macOS, Windows
- Code quality checks
- Security scans

**☕ Take a coffee break while tests run!**

**When all green ✅ → Continue to Step 6**

---

### Step 6: Publish to PyPI (2 minutes)

**Run the publish script:**

```cmd
cd "C:\Users\Lenovo i7\Desktop\my_personal_projects\chunckerflow"

publish_quick.bat
```

**Expected output:**

```
===================================
ChunkFlow Quick Publish Script
===================================

Working directory: C:\Users\Lenovo i7\...\chunckerflow

[1/4] Cleaning old builds...
    Done.

[2/4] Cleaning Jupyter notebooks...
✓ Cleaned 01_getting_started.ipynb
✓ Cleaned 02_strategy_comparison.ipynb
✓ Cleaned 03_advanced_metrics.ipynb
✓ Cleaned 04_visualization_analysis.ipynb
✓ Cleaned 05_api_usage.ipynb
    Done.

[3/4] Building distribution packages...
    Build complete.

[4/4] Validating distribution packages...
    Validation passed.

===================================
Ready to upload to PyPI
===================================

Files to upload:
chunk-flow-0.1.0.tar.gz
chunk_flow-0.1.0-py3-none-any.whl

Checking for .pypirc configuration...
  ✓ Found .pypirc - will use stored credentials
  No password prompt needed!

Press any key to continue with upload...
```

**Press any key to upload!**

**If successful, you'll see:**
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading chunk-flow-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading chunk_flow-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://pypi.org/project/chunk-flow/0.1.0/

===================================
✓ SUCCESS! Package published to PyPI
===================================
```

🎉 **You're live on PyPI!**

**Verify:** Go to https://pypi.org/project/chunk-flow/

---

### Step 7: Create Git Tag (1 minute)

```cmd
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

**Expected output:**
```
Total 0 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/YOUR-USERNAME/chunk-flow.git
 * [new tag]         v0.1.0 -> v0.1.0
```

---

### Step 8: Create GitHub Release (3 minutes)

**Go to:** https://github.com/YOUR-USERNAME/chunk-flow/releases/new

**Fill in:**

1. **Choose a tag:** Select `v0.1.0`

2. **Release title:**
   ```
   ChunkFlow v0.1.0
   ```

3. **Description:** Copy from CHANGELOG.md (lines 10-90)

4. **Attach files:** Upload these from `dist/` folder:
   - `chunk-flow-0.1.0.tar.gz`
   - `chunk_flow-0.1.0-py3-none-any.whl`

5. **Click:** "Publish release"

✅ **Done!**

---

### Step 9: Test Installation (2 minutes)

**Test that others can install your package:**

```cmd
REM Create a test environment
python -m venv test_install
test_install\Scripts\activate

REM Install from PyPI
pip install chunk-flow

REM Test it works
python -c "from chunk_flow.chunking import StrategyRegistry; print('✓ ChunkFlow works!')"

REM Clean up
deactivate
rmdir /s /q test_install
```

**If you see `✓ ChunkFlow works!` → SUCCESS!** 🎉

---

### Step 10: Update to Dev Version (2 minutes)

**Prepare for next release:**

**Edit `chunk_flow/__init__.py`:**
```python
__version__ = "0.2.0.dev0"
```

**Edit `pyproject.toml`:**
```toml
version = "0.2.0.dev0"
```

**Commit:**
```cmd
git add chunk_flow/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.2.0.dev0"
git push origin main
```

---

## 🎉 Congratulations!

**Your package is live!**

### Check Your Package

- **PyPI:** https://pypi.org/project/chunk-flow/
- **GitHub:** https://github.com/YOUR-USERNAME/chunk-flow/releases
- **Stats:** https://pypistats.org/packages/chunk-flow

### Install Command

Anyone can now install your package:
```bash
pip install chunk-flow
```

### Share the News!

**Tweet/Post:**
```
🚀 Just published ChunkFlow v0.1.0 to PyPI!

Production-grade text chunking framework for RAG systems:
- 6 chunking strategies
- 12 evaluation metrics
- Async-first design
- Ready for production

pip install chunk-flow

#Python #RAG #AI #OpenSource
```

**LinkedIn/Blog:**
```
I'm excited to announce the release of ChunkFlow v0.1.0! 🚀

ChunkFlow is a comprehensive framework for text chunking in RAG systems,
featuring 6 strategies, 12 metrics, and production-grade practices.

Check it out: https://pypi.org/project/chunk-flow/
GitHub: https://github.com/YOUR-USERNAME/chunk-flow
```

---

## ⚠️ Troubleshooting

### Issue: "Invalid credentials"

**Fix:**
```cmd
REM Check .pypirc exists
dir %USERPROFILE%\.pypirc

REM View contents
type %USERPROFILE%\.pypirc

REM Make sure format is:
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

### Issue: "Version already exists"

**Fix:**
```
You can't re-upload the same version.
Increment version to 0.1.1 and rebuild.
```

### Issue: "Package not found on PyPI"

**Wait:** It can take 1-2 minutes for PyPI to index new packages.
Refresh the page.

### Issue: GitHub Actions failed

**Fix:**
```
Don't publish if tests fail!
Check the Actions tab to see what failed.
Fix the issues, commit, push, and try again.
```

---

## 📊 Post-Release Analytics

**Track your package:**

1. **Downloads:** https://pypistats.org/packages/chunk-flow
2. **GitHub Stars:** Check your repo
3. **PyPI Stats:** https://pypi.org/project/chunk-flow/

**Badges will update automatically in your README!**

---

## 🔄 Next Release

**For version 0.1.1 (bug fix):**
1. Make your fixes
2. Update version to `0.1.1`
3. Update CHANGELOG.md
4. Repeat steps 3-10

**For version 0.2.0 (new features):**
1. Develop new features
2. Update version to `0.2.0`
3. Update CHANGELOG.md
4. Repeat steps 3-10

---

## 📚 Quick Command Reference

```cmd
REM Update .pypirc
notepad %USERPROFILE%\.pypirc

REM Commit changes
git add .
git commit -m "chore: prepare release v0.1.0"
git push origin main

REM Publish
publish_quick.bat

REM Create tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

REM Test install
pip install chunk-flow
```

---

## ✅ Final Checklist

Before you start, make sure:

- [x] Project code is complete
- [x] Tests are passing (locally or on GitHub Actions)
- [x] Documentation is up to date
- [x] Examples work correctly
- [x] Old PyPI token is revoked
- [x] New token is in .pypirc
- [x] You have a PyPI account
- [x] You're ready to share with the world!

---

## 🎯 Time Estimate

**Total time:** ~20-30 minutes

- Step 1-2: Version & Changelog (5 min)
- Step 3-4: Commit & Push (3 min)
- Step 5: Wait for CI (5 min) ☕
- Step 6: Publish (2 min)
- Step 7-8: Tag & Release (4 min)
- Step 9-10: Test & Cleanup (4 min)

**Most of the time is waiting for GitHub Actions!**

---

**Ready? Let's do this!** 🚀

**Start with Step 1 above!**

Good luck with your release! 🎉
