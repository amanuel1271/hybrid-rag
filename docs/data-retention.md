# Data Retention Policy

## Shipment records
Shipment records, including addresses and proof-of-delivery photos, are kept for 5 years because of customs and tax requirements, then deleted by the nightly retention job.

## Logs
Application logs are kept for 30 days in hot storage and 1 year in cold storage. Logs must never contain full card numbers, passwords or government ID numbers; the log scrubber masks anything that looks like a card number before ingestion.

## Customer deletion requests
When a customer asks for their personal data to be deleted, the privacy team has 30 days to complete the request. Shipment records required for customs are kept but the personal fields are anonymized.
