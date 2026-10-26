# Database Backups

## Schedule
The primary PostgreSQL cluster takes a full snapshot every night at 03:00 KST and ships write-ahead logs continuously, which allows point-in-time recovery to any second in the last 14 days.

## Restore drills
The platform team runs a restore drill on the first Tuesday of every month, restoring the latest snapshot into an isolated environment and checking row counts. The target is to restore the booking database in under 45 minutes.

## Retention
Nightly snapshots are kept for 35 days. Monthly snapshots are kept for 1 year.
