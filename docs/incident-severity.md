# Incident Severity Levels

## SEV1
A SEV1 is a full outage of shipment tracking or the booking API, or any incident with confirmed customer data exposure. SEV1 requires an incident commander within 15 minutes, a status page update every 30 minutes, and a written postmortem within 3 business days.

## SEV2
A SEV2 is a partial outage or severe degradation: for example p99 latency of the booking API above 2 seconds for more than 10 minutes, or one region unavailable. Status page updates are hourly. A postmortem is required within 5 business days.

## SEV3
A SEV3 is a minor bug with a workaround. No status page update is needed and no postmortem is required, but a ticket must be filed in the RELIABILITY project.

## Postmortems
Postmortems are blameless. Every postmortem must list action items with an owner and a due date, and the action items are reviewed in the Thursday reliability meeting.
