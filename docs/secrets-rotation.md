# Secrets and Key Rotation

## Storage
All production secrets live in the secrets manager. Secrets must never be committed to git; the pre-commit hook and the CI secret scanner block commits that contain credentials.

## Rotation schedule
Database passwords are rotated every 90 days. Carrier API keys are rotated every 180 days. TLS certificates are renewed automatically 30 days before expiry.

## Leaked credentials
If a credential is leaked, treat it as a SEV1: revoke it immediately, rotate it, and search the audit log for any use of the leaked credential in the last 30 days.
