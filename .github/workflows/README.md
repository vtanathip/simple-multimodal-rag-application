# GitHub Actions CI/CD Badges

Add these badges to your README.md file to show the current status of your tests:

```markdown
[![CI/CD Pipeline](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/ci.yml/badge.svg)](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/ci.yml)
[![Test Suite](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/test.yml/badge.svg)](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/test.yml)
[![Multimodal RAG Tests](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/multimodal-rag-tests.yml/badge.svg)](https://github.com/vtanathip/simple-multimodal-rag-application/actions/workflows/multimodal-rag-tests.yml)
```

## Workflow Overview

The repository now includes the following GitHub Actions workflows:

### 1. `ci.yml` - Comprehensive CI/CD Pipeline

- Runs on push to main, develop, openwebui branches and PRs
- Tests with multiple Python versions
- Includes linting, type checking, security scans
- Builds and tests Docker images
- Runs integration tests with Milvus service

### 2. `test.yml` - Simple Test Runner

- Focused on running pytest only
- Lightweight and fast
- Good for basic PR checks

### 3. `multimodal-rag-tests.yml` - Specialized Test Suite

- Comprehensive testing for the multimodal RAG application
- Cross-platform testing (Ubuntu, Windows, macOS)
- Coverage reporting
- Scheduled daily runs to catch dependency issues
- Docker container testing

### 4. `pr-tests.yml` - Pull Request Tests

- Runs fast tests on PR creation/updates
- Adds automatic comments to PRs with test results
- Skips draft PRs

## Local Testing

Before pushing, you can run the same tests locally:

```bash
# Run all tests
uv run pytest test/ -v

# Run with coverage
uv add pytest-cov --dev
uv run pytest test/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test files
uv run pytest test/test_api.py -v
```