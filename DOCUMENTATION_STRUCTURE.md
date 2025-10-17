# ChunkFlow Documentation Structure

This document explains the organization of ChunkFlow's documentation.

## 📁 Directory Structure

```
chunckerflow/
├── 📄 README.md                    # Main project README
├── 📄 CHANGELOG.md                 # Version history
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 RELEASE.md                   # Release process guide
├── 📄 DOCKER.md                    # Docker deployment guide
├── 📄 LICENSE                      # MIT License
│
├── 🔧 pyproject.toml               # Python project configuration
├── 🔧 Makefile                     # Build automation
├── 🔧 Dockerfile                   # Docker image definition
├── 🔧 docker-compose.yml           # Docker compose setup
├── 🔧 .gitignore                   # Git ignore rules
│
├── 🚀 publish.bat                  # Publish script (with tests)
├── 🚀 publish_quick.bat            # Quick publish script (no tests)
├── 🧹 clean_notebooks.py           # Jupyter notebook cleaner
│
├── 📚 docs/                        # All documentation
│   ├── README.md                   # Documentation index
│   │
│   ├── 🚀 Publishing Guides
│   │   ├── BUILD_AND_PUBLISH.md   # Complete publishing guide
│   │   ├── QUICK_PUBLISH.txt      # Quick reference card
│   │   └── PUBLISH_FIX_SUMMARY.md # Recent fixes
│   │
│   ├── 🔒 Security Guides
│   │   ├── SECURITY_WARNING.md    # Token security (CRITICAL)
│   │   └── TOKEN_SETUP_COMPLETE.md# .pypirc setup guide
│   │
│   ├── 📋 Project Information
│   │   ├── PROJECT_SUMMARY.md     # Complete project overview
│   │   └── FINALIZATION_SUMMARY.md# Recent improvements
│   │
│   └── 📖 User Guides (existing)
│       ├── GETTING_STARTED.md
│       ├── API_REFERENCE.md
│       ├── guides/
│       ├── concepts/
│       ├── api-reference/
│       └── tutorials/
│
├── 💻 chunk_flow/                  # Source code
│   ├── __init__.py
│   ├── core/
│   ├── chunking/
│   ├── embeddings/
│   ├── evaluation/
│   ├── analysis/
│   ├── api/
│   └── utils/
│
├── 🧪 tests/                       # Test suite
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── 📊 benchmarks/                  # Performance benchmarks
│   └── run_benchmarks.py
│
└── 📓 examples/                    # Usage examples
    ├── basic_usage.py
    ├── strategy_comparison.py
    ├── analysis_and_visualization.py
    ├── api_client_example.py
    │
    └── jupyter/                    # Interactive notebooks
        ├── README.md
        ├── 01_getting_started.ipynb
        ├── 02_strategy_comparison.ipynb
        ├── 03_advanced_metrics.ipynb
        ├── 04_visualization_analysis.ipynb
        └── 05_api_usage.ipynb
```

---

## 📚 Documentation Categories

### 🏠 Root Level (User-Facing)

Essential files that users need immediately:

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Project overview, quick start | Everyone |
| **CHANGELOG.md** | Version history | Users, contributors |
| **CONTRIBUTING.md** | How to report issues | Users wanting to help |
| **RELEASE.md** | Release process | Maintainers |
| **DOCKER.md** | Docker deployment | DevOps, deployment |
| **LICENSE** | MIT License terms | Everyone |

### 📚 docs/ Directory (Comprehensive Guides)

Detailed documentation organized by topic:

#### 🚀 Publishing (for Maintainers)

| File | Purpose | When to Read |
|------|---------|--------------|
| **BUILD_AND_PUBLISH.md** | Complete PyPI publishing guide | Before first publish |
| **QUICK_PUBLISH.txt** | Quick reference for publishing | Every publish |
| **PUBLISH_FIX_SUMMARY.md** | Recent script fixes | If having issues |

#### 🔒 Security (CRITICAL)

| File | Purpose | When to Read |
|------|---------|--------------|
| **SECURITY_WARNING.md** | Token security best practices | **IMMEDIATELY** |
| **TOKEN_SETUP_COMPLETE.md** | .pypirc configuration | Before publishing |

#### 📋 Project Info

| File | Purpose | When to Read |
|------|---------|--------------|
| **PROJECT_SUMMARY.md** | Complete project overview | To understand project |
| **FINALIZATION_SUMMARY.md** | Recent improvements | After updates |

#### 📖 User Guides

| File | Purpose | Audience |
|------|---------|----------|
| **GETTING_STARTED.md** | Installation and first steps | New users |
| **API_REFERENCE.md** | Complete API documentation | Developers |
| **guides/** | Topic-specific guides | All users |
| **concepts/** | Conceptual explanations | Learning users |
| **api-reference/** | Detailed API docs | Developers |
| **tutorials/** | Step-by-step tutorials | New users |

---

## 🎯 Quick Navigation

### I want to...

#### **Get started with ChunkFlow**
→ Start: [README.md](../README.md)
→ Then: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
→ Try: [examples/jupyter/01_getting_started.ipynb](examples/jupyter/01_getting_started.ipynb)

#### **Publish to PyPI**
→ **FIRST:** [docs/SECURITY_WARNING.md](docs/SECURITY_WARNING.md) ⚠️
→ Setup: [docs/TOKEN_SETUP_COMPLETE.md](docs/TOKEN_SETUP_COMPLETE.md)
→ Publish: [docs/QUICK_PUBLISH.txt](docs/QUICK_PUBLISH.txt)
→ Details: [docs/BUILD_AND_PUBLISH.md](docs/BUILD_AND_PUBLISH.md)

#### **Deploy with Docker**
→ Guide: [DOCKER.md](../DOCKER.md)
→ Compose: [docker-compose.yml](../docker-compose.yml)

#### **Understand the codebase**
→ Overview: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)
→ API: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

#### **Report a bug or request feature**
→ Guidelines: [CONTRIBUTING.md](../CONTRIBUTING.md)
→ Issues: https://github.com/chunkflow/chunk-flow/issues

---

## 📖 Recommended Reading Paths

### For New Users

1. [README.md](../README.md) - Project overview
2. [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - Installation
3. [examples/jupyter/01_getting_started.ipynb](examples/jupyter/01_getting_started.ipynb) - Interactive tutorial
4. [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - API details

### For Contributors/Maintainers

1. [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) - Project structure
2. [CONTRIBUTING.md](../CONTRIBUTING.md) - Guidelines
3. [docs/SECURITY_WARNING.md](docs/SECURITY_WARNING.md) - Security ⚠️
4. [docs/BUILD_AND_PUBLISH.md](docs/BUILD_AND_PUBLISH.md) - Publishing

### For Deployment

1. [DOCKER.md](../DOCKER.md) - Docker guide
2. [RELEASE.md](../RELEASE.md) - Release process
3. [docs/SECURITY_WARNING.md](docs/SECURITY_WARNING.md) - Token security

---

## 🔍 Finding Documentation

### By File Extension

- **`.md`** - Markdown documentation (most docs)
- **`.txt`** - Plain text quick reference
- **`.ipynb`** - Interactive Jupyter notebooks
- **`.py`** - Python code examples

### By Location

- **Root** - Essential user-facing docs
- **docs/** - Comprehensive guides and references
- **examples/** - Working code examples
- **examples/jupyter/** - Interactive tutorials

### By Purpose

- **Learning** → examples/jupyter/*.ipynb
- **Reference** → docs/API_REFERENCE.md
- **Publishing** → docs/BUILD_AND_PUBLISH.md
- **Security** → docs/SECURITY_WARNING.md
- **Deployment** → DOCKER.md

---

## 📋 Documentation Maintenance

### Adding New Documentation

1. **User guides** → Place in `docs/guides/`
2. **API references** → Place in `docs/api-reference/`
3. **Tutorials** → Place in `docs/tutorials/` or `examples/jupyter/`
4. **Root-level** → Only for critical user-facing docs

### Updating Documentation

1. Update the specific doc file
2. Update [docs/README.md](docs/README.md) index if needed
3. Update this file if structure changes
4. Update main [README.md](../README.md) if user-facing changes

### Documentation Standards

- Use **GitHub Flavored Markdown**
- Include **table of contents** for long docs
- Use **relative links** for cross-references
- Keep **line length ≤ 100** characters
- Use **clear headings** and structure
- Include **code examples** where relevant

---

## 🎨 Documentation Style Guide

### Emojis for Navigation

Use consistently:
- 📚 Documentation/Guides
- 🚀 Publishing/Deployment
- 🔒 Security
- 💻 Code/Development
- 🧪 Testing
- 📊 Benchmarks/Analysis
- 📓 Examples/Tutorials
- ⚠️ Warnings/Critical
- ✅ Success/Completed
- 📋 Lists/Checklists
- 🎯 Goals/Objectives
- 💡 Tips/Hints

### Heading Hierarchy

```markdown
# Main Title (H1) - Use once per document
## Major Section (H2)
### Subsection (H3)
#### Detail (H4)
```

### Code Blocks

````markdown
```python
# Python code
def example():
    pass
```

```bash
# Shell commands
python script.py
```

```cmd
REM Windows batch
echo Hello
```
````

---

## 📊 Documentation Stats

### Current Documentation

- **Total Docs**: 10+ markdown files in docs/
- **Interactive Tutorials**: 5 Jupyter notebooks
- **Code Examples**: 4 Python examples
- **Root Docs**: 6 essential files
- **Lines of Documentation**: 5,000+ lines

### Coverage

- ✅ Installation guide
- ✅ Quick start
- ✅ API reference
- ✅ Publishing guide
- ✅ Security guide
- ✅ Deployment guide
- ✅ Interactive tutorials
- ✅ Code examples

---

## 🔗 External Documentation

- **PyPI Package**: https://pypi.org/project/chunk-flow/
- **GitHub Repository**: https://github.com/chunkflow/chunk-flow
- **Issue Tracker**: https://github.com/chunkflow/chunk-flow/issues
- **Discussions**: https://github.com/chunkflow/chunk-flow/discussions

---

## ✅ Quick Reference Card

### Essential Files

```
📄 README.md              # Start here!
📄 docs/README.md         # Documentation index
📄 docs/GETTING_STARTED.md# Installation guide
📄 docs/SECURITY_WARNING.md# Security (CRITICAL)
📄 docs/BUILD_AND_PUBLISH.md# Publishing guide
```

### Quick Commands

```bash
# Read main README
cat README.md

# Browse documentation
cd docs && ls -la

# View publishing guide
cat docs/QUICK_PUBLISH.txt

# View security warning
cat docs/SECURITY_WARNING.md
```

---

**Documentation last updated:** ChunkFlow v0.1.0 preparation

For the latest updates, check git history or CHANGELOG.md.

**Need help? Start with [docs/README.md](docs/README.md)!** 📚
