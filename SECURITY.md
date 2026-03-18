# Security and Version Updates

## Recent Changes

### Dependency Security Updates (Latest)
- Updated `langchain-community` from 0.0.20 to 0.3.27
  - Fixes XXE vulnerability (CVE affecting versions < 0.3.27)
  - Fixes SSRF vulnerability in RequestsToolkit (CVE affecting versions < 0.0.28)
  - Fixes pickle deserialization vulnerability (CVE affecting versions < 0.2.4)
- Updated `langchain` from 0.1.6 to 0.3.27 for compatibility
- Updated `langchain-openai` from 0.0.5 to 0.2.10 for compatibility

### Compatibility Notes
The code has been tested and is compatible with:
- Python 3.8+
- Latest LangChain versions (0.3.x)
- OpenAI API (latest)

### Breaking Changes
None - All APIs used in this project remain stable across the version updates.

## Security Best Practices

1. **Keep Dependencies Updated**: Regularly run `pip list --outdated` to check for updates
2. **API Key Security**: Never commit `.env` files or expose API keys
3. **Input Validation**: The system validates user inputs to prevent injection attacks
4. **Sandbox PDFs**: Only process PDFs from trusted sources

## Updating Dependencies

To update all dependencies to their latest secure versions:

```bash
pip install --upgrade -r requirements.txt
```

To check for security vulnerabilities:

```bash
pip install safety
safety check
```
