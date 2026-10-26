# On-Call Runbook

## Rotation
Orbit Freight runs a weekly on-call rotation that starts every Monday at 10:00 KST. Each rotation has a primary and a secondary engineer. The primary must acknowledge a page within 5 minutes; if they do not, PagerDuty escalates to the secondary after 10 minutes and to the engineering manager after 20 minutes.

## Handover
The outgoing primary writes a handover note in the #oncall channel before 10:00 on Monday. The note lists open incidents, silenced alerts and anything deployed on Friday. Silenced alerts must have an expiry; an alert may not be silenced for longer than 72 hours.

## Compensation
On-call engineers receive a flat stipend of 150,000 KRW per week, plus 1 day of time off in lieu for every weekend page that took longer than 2 hours to resolve.
