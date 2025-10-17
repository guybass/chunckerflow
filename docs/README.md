# ChunkFlow Documentation

Welcome to the ChunkFlow documentation! This directory contains comprehensive guides for using, publishing, and understanding the ChunkFlow framework.

## 📚 Documentation Index

### 🚀 Publishing Guides

| File | Description | When to Read |
|------|-------------|--------------|
| **[BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)** | Complete guide to building and publishing ChunkFlow to PyPI | Before first publish |
| **[QUICK_PUBLISH.txt](QUICK_PUBLISH.txt)** | Quick reference card for publishing | Every publish |
| **[PUBLISH_FIX_SUMMARY.md](PUBLISH_FIX_SUMMARY.md)** | Recent fixes to publish scripts | If having issues |

### 🔒 Security Guides

| File | Description | When to Read |
|------|-------------|--------------|
| **[SECURITY_WARNING.md](SECURITY_WARNING.md)** | Critical security information about API tokens | **READ IMMEDIATELY** |
| **[TOKEN_SETUP_COMPLETE.md](TOKEN_SETUP_COMPLETE.md)** | PyPI token setup and .pypirc configuration | After reading security warning |

### 📋 Project Information

| File | Description | When to Read |
|------|-------------|--------------|
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | Complete project overview and phase accomplishments | To understand project structure |
| **[FINALIZATION_SUMMARY.md](FINALIZATION_SUMMARY.md)** | Repository finalization and cleanup summary | To understand recent changes |

---

## 🎯 Quick Navigation

### I want to...

#### **Publish to PyPI**
1. Read: [SECURITY_WARNING.md](SECURITY_WARNING.md) ⚠️ **CRITICAL FIRST**
2. Setup: [TOKEN_SETUP_COMPLETE.md](TOKEN_SETUP_COMPLETE.md)
3. Publish: [QUICK_PUBLISH.txt](QUICK_PUBLISH.txt)
4. Details: [BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)

#### **Understand the project**
1. Overview: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Recent changes: [FINALIZATION_SUMMARY.md](FINALIZATION_SUMMARY.md)

#### **Fix publishing issues**
1. Check: [PUBLISH_FIX_SUMMARY.md](PUBLISH_FIX_SUMMARY.md)
2. Reference: [BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)

#### **Secure my tokens**
1. **URGENT:** [SECURITY_WARNING.md](SECURITY_WARNING.md)
2. Setup: [TOKEN_SETUP_COMPLETE.md](TOKEN_SETUP_COMPLETE.md)

---

## 📖 Recommended Reading Order

### For First-Time Publishers

1. 🔒 **[SECURITY_WARNING.md](SECURITY_WARNING.md)** - Learn about token security (**CRITICAL**)
2. 🔐 **[TOKEN_SETUP_COMPLETE.md](TOKEN_SETUP_COMPLETE.md)** - Set up .pypirc
3. 📚 **[BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)** - Understand the full process
4. ⚡ **[QUICK_PUBLISH.txt](QUICK_PUBLISH.txt)** - Use as quick reference
5. 🚀 Publish using `publish_quick.bat`!

### For Understanding the Project

1. 📋 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - See what was built
2. ✨ **[FINALIZATION_SUMMARY.md](FINALIZATION_SUMMARY.md)** - See recent improvements
3. 🔧 **[PUBLISH_FIX_SUMMARY.md](PUBLISH_FIX_SUMMARY.md)** - See bug fixes

---

## 📂 Additional Documentation

### In Project Root

- **README.md** - Main project README with quick start
- **CHANGELOG.md** - Version history and changes
- **CONTRIBUTING.md** - How to report issues and request features
- **RELEASE.md** - Release process guide
- **LICENSE** - MIT License

### In Examples Directory

- **examples/jupyter/** - Interactive Jupyter notebook tutorials
  - 01_getting_started.ipynb
  - 02_strategy_comparison.ipynb
  - 03_advanced_metrics.ipynb
  - 04_visualization_analysis.ipynb
  - 05_api_usage.ipynb

---

## 🚨 Critical Reading

### ⚠️ MUST READ Before Publishing

**[SECURITY_WARNING.md](SECURITY_WARNING.md)** - Contains critical information about:
- API token security
- How to revoke compromised tokens
- Setting up .pypirc securely
- Security best practices

**If you've shared your PyPI token anywhere, read this immediately!**

---

## 💡 Quick Tips

### Publishing Workflow

```cmd
REM 1. Update version and changelog
REM 2. Commit and push to GitHub
git push origin main

REM 3. Wait for GitHub Actions to pass
REM 4. Run publish script
publish_quick.bat

REM 5. Create tag and release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### Token Security

```cmd
REM Store token in .pypirc (NOT in code!)
notepad %USERPROFILE%\.pypirc

REM Add:
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

### Getting Help

- **Publishing issues:** See [PUBLISH_FIX_SUMMARY.md](PUBLISH_FIX_SUMMARY.md)
- **Security concerns:** See [SECURITY_WARNING.md](SECURITY_WARNING.md)
- **General questions:** See [BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)

---

## 📞 Support

If you need help:

1. **Check this documentation first** - Most questions are answered here
2. **GitHub Issues** - Report bugs or request features
3. **GitHub Discussions** - Ask questions

---

## 🔗 External Resources

- **PyPI Help:** https://pypi.org/help/
- **Python Packaging Guide:** https://packaging.python.org/
- **Twine Documentation:** https://twine.readthedocs.io/
- **PyPI Token Guide:** https://pypi.org/help/#apitoken

---

## 📋 Documentation Checklist

Before publishing, make sure you've read:

- [ ] **SECURITY_WARNING.md** - Token security ⚠️
- [ ] **TOKEN_SETUP_COMPLETE.md** - .pypirc setup
- [ ] **QUICK_PUBLISH.txt** - Publishing steps
- [ ] Your .pypirc is configured correctly
- [ ] Your old token (if any) is revoked
- [ ] You understand the workflow

---

## ✨ Last Updated

This documentation was created as part of the ChunkFlow v0.1.0 release preparation.

For the latest updates, check the git history or CHANGELOG.md in the project root.

---

**Ready to publish? Start with [SECURITY_WARNING.md](SECURITY_WARNING.md)!** 🚀
