# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in the CairnIQ, please report it responsibly.

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge receipt within 48 hours and work on a fix.

## Security Design Principles

### Local-First Architecture
- All user data is stored locally in `user_data/`
- API keys are never transmitted to any central server
- No telemetry or analytics are collected

### Data Protection
- The `user_data/` directory is excluded from version control via `.gitignore`
- The packaging system explicitly strips all personal data from distributions
- Brokerage credentials are stored only in the local `.env` file

### API Key Safety
- Keys are masked in the Settings UI (only first/last 4 characters shown)
- The `.env` file is never included in distribution packages
- All API calls use HTTPS

### Network Exposure
- The server binds to `127.0.0.1` by default and is intended for local use only
- Do **not** expose the server to a public network (`CAIRNIQ_HOST=0.0.0.0`) — the web-reader tool fetches arbitrary URLs supplied by the AI, which could be exploited for SSRF in a multi-user or network-accessible deployment

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes    |
| 1.x     | ❌ No     |

## Best Practices for Users

1. **Never commit** your `user_data/.env` file to version control
2. **Rotate** API keys periodically
3. **Use paper trading** mode when testing brokerage integrations
4. **Back up** your `user_data/` directory regularly
