# Contributing to CairnIQ

Thank you for your interest in contributing! This project is a personal wealth management tool and we welcome improvements that enhance its reliability, security, and usability.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Install** dependencies:
   ```bash
   ./install.sh
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional but recommended) Dev tooling + secret-scanning git hook
pip install -r requirements-dev.txt
pre-commit install

# Run the server
python server.py
```

## Code Guidelines

- **Python 3.12+** is required
- Use the existing logging system (`agent/logger.py`) — avoid raw `print()` statements
- All user data must be stored in `user_data/` — never commit personal data
- Wrap external API calls with timeouts and error handling
- Add tests for new functionality in `tests/`

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Ensure your code passes all existing tests
2. Add tests for new functionality
3. Update documentation if needed
4. Submit a pull request with a clear description of the changes

## What We're Looking For

- Bug fixes and stability improvements
- New financial data integrations
- UI/UX improvements
- Documentation improvements
- Test coverage improvements

## What We Won't Accept

- Changes that transmit user data externally
- Commercial or enterprise features (see LICENSE)
- Dependencies with incompatible licenses

## Questions?

Open an issue for discussion before starting large changes.
