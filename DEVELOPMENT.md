# Development-Guidelines
## Pre-commit Hooks
Before committing any changes, make sure to enable the pre-commit hooks.
This will help you to automatically format your code and check for any linting issues.
You can enable the pre-commit hooks by running the following command:
```bash
pre-commit install
```
In case you want to run the pre-commit hooks manually, you can do so by running:
```bash
pre-commit run --all-files
```
> **_NOTE:_** Be aware that the code in its current state does not comply with the pre-commit hooks.
> Hence, you might encounter errors in sections that are not related to your changes.
> This is intended to slowly improve the code quality over time.
> If fixing the errors would cause a massive overhead, you can ignore them via the `--no-verify` flag when committing.