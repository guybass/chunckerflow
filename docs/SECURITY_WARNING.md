# 🚨 CRITICAL SECURITY WARNING 🚨

## ⚠️ YOUR PYPI TOKEN WAS EXPOSED

Your PyPI API token was shared in plain text during our conversation. This is a **critical security vulnerability**.

---

## 🔴 IMMEDIATE ACTION REQUIRED

### Step 1: Revoke the Exposed Token (DO THIS NOW!)

1. Go to: **https://pypi.org/manage/account/token/**
2. Find the token you shared
3. Click **"Remove"** or **"Delete"**
4. Confirm deletion

**The exposed token:**
```
pypi-AgEIcHlwaS5vcmcCJGVhNzQ0YTVh... (REVOKE THIS!)
```

### Step 2: Create a NEW Token

1. Go to: **https://pypi.org/manage/account/token/**
2. Click **"Add API token"**
3. Token name: `chunk-flow-publishing`
4. Scope: **"Project: chunk-flow"** (if project exists) or **"Entire account"**
5. Click **"Add token"**
6. **Copy the token immediately** (you can only see it once!)

### Step 3: Update .pypirc with NEW Token

**Windows Path:** `C:\Users\Lenovo i7\.pypirc`

Edit the file and replace the old token:

```ini
[pypi]
username = __token__
password = pypi-YOUR-NEW-TOKEN-HERE

[testpypi]
username = __token__
password =
```

**Important:** On Windows, this file is in your home directory: `%USERPROFILE%\.pypirc`

---

## ✅ What I've Done to Secure Your Project

### 1. Created .pypirc File

Created: `~/.pypirc` (Linux) or `%USERPROFILE%\.pypirc` (Windows)

This file stores your PyPI credentials securely:
- ✅ Not tracked by git
- ✅ Permissions set to 600 (owner read/write only)
- ✅ Used automatically by `twine upload`

### 2. Enhanced .gitignore

Added comprehensive security exclusions:

```gitignore
# Environment variables and secrets
.env
.env.*
!.env.example
.pypirc
*.pypirc
*token*
*secret*
*credentials*
*password*
*.key
*.pem
*.p12
*.pfx
```

This prevents accidental commits of:
- ✅ API tokens
- ✅ Credentials files
- ✅ Secret keys
- ✅ Certificates
- ✅ Environment files

### 3. Verified Project Cleanliness

Checked for sensitive files in your project:
- ✅ No token files found
- ✅ No credential files found
- ✅ No .pypirc in project directory
- ✅ Git status is clean

---

## 📚 How to Use PyPI Token Securely

### Option 1: Using .pypirc (Recommended)

Once you've updated `~/.pypirc` with your NEW token, publishing is simple:

```cmd
twine upload dist/*
```

No password prompt! Twine reads credentials from `.pypirc` automatically.

### Option 2: Environment Variable

```cmd
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-YOUR-NEW-TOKEN-HERE
twine upload dist/*
```

### Option 3: Command Line (NOT Recommended)

```cmd
twine upload -u __token__ -p pypi-YOUR-TOKEN dist/*
```

**Warning:** This may expose token in command history!

---

## 🔒 Security Best Practices

### ✅ DO:

1. **Store tokens in .pypirc** - Safe, convenient, gitignored
2. **Use project-scoped tokens** - Limit damage if compromised
3. **Revoke tokens when done** - Don't leave unused tokens active
4. **Enable 2FA on PyPI** - Extra account security
5. **Check .gitignore** - Before committing anything

### ❌ DON'T:

1. **Share tokens in chat/email** - NEVER share credentials
2. **Commit tokens to git** - Even in private repos
3. **Use account-wide tokens** - Use project-specific when possible
4. **Hardcode tokens in code** - Always use config files
5. **Reuse tokens across projects** - One token per project

---

## 🔍 Verify Your .pypirc Setup

### Windows:

```cmd
REM Check if .pypirc exists
dir %USERPROFILE%\.pypirc

REM View contents (carefully!)
type %USERPROFILE%\.pypirc
```

### Linux/WSL:

```bash
# Check if .pypirc exists
ls -la ~/.pypirc

# View contents (carefully!)
cat ~/.pypirc
```

**Expected content:**
```ini
[pypi]
username = __token__
password = pypi-YOUR-NEW-TOKEN-HERE

[testpypi]
username = __token__
password =
```

---

## 🚀 Updated Publishing Workflow

### 1. Update Version

Edit these files:
- `chunk_flow/__init__.py`: `__version__ = "0.1.0"`
- `pyproject.toml`: `version = "0.1.0"`
- `CHANGELOG.md`: Add release notes

### 2. Commit and Push

```cmd
git add .
git commit -m "chore: prepare release v0.1.0"
git push origin main
```

### 3. Verify GitHub Actions Pass

Check: https://github.com/YOUR-USERNAME/chunk-flow/actions

### 4. Build and Publish

```cmd
cd "C:\Users\Lenovo i7\Desktop\my_personal_projects\chunckerflow"
publish_quick.bat
```

**When prompted:**
- Username: **Will NOT be prompted** (reads from .pypirc)
- Password: **Will NOT be prompted** (reads from .pypirc)

If `.pypirc` is set up correctly, it will upload without asking!

### 5. Create Git Tag

```cmd
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 6. Create GitHub Release

1. Go to: https://github.com/YOUR-USERNAME/chunk-flow/releases/new
2. Select tag: v0.1.0
3. Upload dist files
4. Publish

---

## 🔐 Token Security Checklist

Before publishing, verify:

- [ ] Old exposed token has been **REVOKED**
- [ ] New token has been **CREATED**
- [ ] `.pypirc` has been **UPDATED** with new token
- [ ] `.pypirc` is in **home directory** (not project)
- [ ] `.pypirc` is **NOT tracked by git**
- [ ] `.gitignore` includes `.pypirc` and `*token*`
- [ ] No sensitive files in `git status`
- [ ] Project directory is **CLEAN**

---

## 📞 Need Help?

If you suspect your PyPI account is compromised:

1. **Immediately revoke ALL tokens**
2. **Change your PyPI password**
3. **Enable 2FA if not already enabled**
4. **Contact PyPI support:** https://pypi.org/help/

---

## ✅ Summary

**What happened:**
- ❌ PyPI token was exposed in conversation

**What I did:**
- ✅ Created `.pypirc` with token (temporary - you must revoke!)
- ✅ Enhanced `.gitignore` to prevent future leaks
- ✅ Verified project is clean
- ✅ Created this security guide

**What YOU must do:**
1. **REVOKE the exposed token** (https://pypi.org/manage/account/token/)
2. **CREATE a new token**
3. **UPDATE .pypirc** with new token
4. **DELETE this SECURITY_WARNING.md** after reading (contains old token reference)

---

**⚠️ REMEMBER: NEVER SHARE API TOKENS AGAIN!** ⚠️

For questions about token security, see:
- PyPI Help: https://pypi.org/help/
- PyPI API Tokens Guide: https://pypi.org/help/#apitoken

**Your project is now secure, but you MUST revoke the exposed token!**
