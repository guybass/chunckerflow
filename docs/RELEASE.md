# Release Guide

## Pre-Release Checklist

- [ ] All tests passing (`make test`)
- [ ] Code formatted (`make format`)
- [ ] Linting clean (`make lint`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `chunk_flow/__init__.py` and `pyproject.toml`
- [ ] Examples tested
- [ ] Docker build successful

## Release Process

### 1. Prepare Release

```bash
# Ensure clean working directory
git status

# Run full CI locally
make ci

# Update version
# Edit chunk_flow/__init__.py and pyproject.toml
# Update CHANGELOG.md with release date
```

### 2. Create Git Tag

```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0"

# Push tag
git push origin v0.1.0
```

### 3. Build Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Check distribution
twine check dist/*
```

### 4. Test PyPI Upload (Optional)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ chunk-flow
```

### 5. Upload to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Or use GitHub Actions (automatic on release)
# Just create a GitHub release and the workflow will handle it
```

### 6. Create GitHub Release

1. Go to https://github.com/chunkflow/chunk-flow/releases/new
2. Select the tag (v0.1.0)
3. Set release title: "ChunkFlow v0.1.0"
4. Copy relevant section from CHANGELOG.md
5. Upload wheel and sdist from `dist/`
6. Publish release

This will automatically trigger:
- PyPI upload (via GitHub Actions)
- Docker image build and push
- Documentation deployment

### 7. Post-Release

```bash
# Update version to next dev version
# E.g., in __init__.py: __version__ = "0.2.0.dev0"

# Commit version bump
git add chunk_flow/__init__.py pyproject.toml
git commit -m "Bump version to 0.2.0.dev0"
git push origin main
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR version** (X.0.0): Incompatible API changes
- **MINOR version** (0.X.0): New functionality, backwards compatible
- **PATCH version** (0.0.X): Bug fixes, backwards compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fix release
- `0.1.0` → `0.2.0`: New features added
- `0.9.0` → `1.0.0`: First stable release
- `1.0.0` → `2.0.0`: Breaking changes

## Hotfix Process

For critical bugs in production:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/0.1.1 v0.1.0

# Make fixes
git commit -m "Fix critical bug"

# Update version and CHANGELOG
# Tag and release
git tag -a v0.1.1 -m "Hotfix v0.1.1"
git push origin v0.1.1

# Merge back to main
git checkout main
git merge hotfix/0.1.1
git push origin main
```

## Docker Image Versioning

Docker images are tagged with:
- `latest` - Latest release
- `X.Y.Z` - Specific version (e.g., `0.1.0`)
- `X.Y` - Minor version (e.g., `0.1`)
- `X` - Major version (e.g., `0`)

Example:
```bash
docker pull chunkflow/chunkflow:latest
docker pull chunkflow/chunkflow:0.1.0
docker pull chunkflow/chunkflow:0.1
```

## PyPI Setup (First Time)

### 1. Create PyPI Account

1. Sign up at https://pypi.org/account/register/
2. Verify email
3. Enable 2FA (recommended)

### 2. Create API Token

1. Go to https://pypi.org/manage/account/token/
2. Create new token for "chunk-flow" project
3. Copy token (starts with `pypi-`)

### 3. Configure GitHub Secrets

Add to repository secrets:
- `PYPI_API_TOKEN`: Your PyPI API token
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

## Docker Hub Setup (First Time)

### 1. Create Docker Hub Account

1. Sign up at https://hub.docker.com/signup
2. Verify email

### 2. Create Repository

1. Create new repository: `chunkflow/chunkflow`
2. Make it public
3. Add description and README

### 3. Create Access Token

1. Go to Account Settings > Security
2. Create new access token
3. Add to GitHub secrets as `DOCKERHUB_TOKEN`

## Troubleshooting

### Build Fails

```bash
# Check dependencies
pip install --upgrade pip build twine

# Clean and rebuild
rm -rf dist/ build/ *.egg-info
python -m build
```

### Upload Fails

```bash
# Check credentials
twine check dist/*

# Verify package name isn't taken
pip search chunk-flow
```

### GitHub Actions Fails

1. Check GitHub Actions logs
2. Verify secrets are set correctly
3. Ensure branch protections allow workflow

### Docker Build Fails

```bash
# Test build locally
docker build -t chunkflow:test .

# Check logs
docker logs <container-id>
```

## Support

- Issues: https://github.com/chunkflow/chunk-flow/issues
- Discussions: https://github.com/chunkflow/chunk-flow/discussions
- Email: maintainers@chunkflow.com
