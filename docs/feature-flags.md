# Feature Flags

## Naming
Feature flags are named team.feature.purpose, for example booking.split-shipments.rollout. Every flag must have an owner and an expiry date.

## Rollout
New features roll out in stages: internal users, then 1% of customers, then 10%, 50% and 100%. Each stage must run for at least 24 hours without a regression before moving on.

## Cleanup
Flags that have been at 100% for more than 30 days must be removed from the code. The flag service sends a weekly reminder to owners of stale flags.
