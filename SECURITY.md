# Security Policy

## Supported Versions

Security updates are provided for the latest version available on the main branch.

| Version | Supported |
|----------|-----------|
| Main branch | ✅ |
| Older versions | ❌ |

Because Imag-Eval is an actively evolving research project, users are encouraged to work from the latest commit whenever possible.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please do **not** open a public GitHub issue.

Instead, report it privately by contacting:

**Ibrahim MOHAMED SEROUIS**
- Email: ibrahim.mohamed-serouis@talan.com

Please include:

- A description of the issue
- Steps to reproduce the problem
- Potential impact
- Relevant logs, screenshots, or proof-of-concept examples
- Any proposed mitigation if available

All reports will be reviewed as quickly as possible.

---

## What Should Be Reported

Examples of vulnerabilities or security concerns include:

### Credential Exposure

- Hardcoded API keys
- Exposed authentication tokens
- Secrets committed to the repository
- Leakage of Azure OpenAI credentials
- Leakage of Hugging Face access tokens

### Dependency Vulnerabilities

- Known vulnerabilities in project dependencies
- Unsafe package configurations
- Dependency substitution or supply-chain attacks

### Arbitrary Code Execution

- Unsafe deserialization
- Command injection vulnerabilities
- Untrusted code execution pathways
- File-system access vulnerabilities

### Data Integrity Risks

- Benchmark manipulation
- Evaluation-result tampering
- Leaderboard submission fraud
- Methods that could artificially inflate reported scores

### Infrastructure Issues

- Misconfigured services
- Insecure deployment configurations
- Unauthorized access risks

---

## What Is Typically Not Considered a Security Issue

The following are generally not considered security vulnerabilities:

- Model hallucinations
- Benchmark disagreements
- Differences in evaluation results across hardware
- Prompt-generation variability caused by proprietary APIs
- Reproducibility limitations associated with third-party model updates
- Performance regressions without a security impact

These issues should instead be reported through GitHub Issues or Discussions.

---

## Responsible Disclosure

Please allow reasonable time for investigation and remediation before publicly disclosing a vulnerability.

We ask researchers and contributors to:

- Avoid accessing data that does not belong to them.
- Avoid modifying repository resources without authorization.
- Avoid attempts to disrupt project services.
- Report findings privately and responsibly.

Good-faith security research is appreciated.

---

## Secrets and Credentials

Contributors should never commit:

- API keys
- Access tokens
- Passwords
- Private certificates
- Proprietary credentials
- `.env` files containing real values

Before opening a Pull Request, verify that all sensitive information has been removed.

A sample configuration file with placeholder values should be used whenever possible.

---

## Leaderboard Integrity

Maintaining benchmark integrity is a core objective of Imag-Eval.

Leaderboard submissions should include:

- Generated images
- Generation seeds
- Relevant inference parameters
- Evaluation methodology

Submitted results may be independently reproduced and verified before publication.

Any attempt to intentionally manipulate, fabricate, or misrepresent benchmark results may result in rejection of the submission.

---

## Third-Party Services

Imag-Eval may rely on external services, models, APIs, datasets, and frameworks.

Users are responsible for:

- Reviewing the security policies of third-party providers.
- Protecting their own credentials.
- Complying with applicable terms of service.
- Following local regulations and organizational policies.

The project maintainers cannot guarantee the security of third-party systems.

---

## Security Best Practices

When running experiments:

- Use isolated virtual environments.
- Keep dependencies updated.
- Store secrets in environment variables.
- Restrict access to API credentials.
- Review external code before execution.
- Validate benchmark outputs before publication.

---

## Acknowledgements

We appreciate responsible security disclosures and constructive reports that help improve the reliability, reproducibility, and integrity of Imag-Eval.
