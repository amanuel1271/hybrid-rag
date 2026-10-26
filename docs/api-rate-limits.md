# Public API Rate Limits

## Tiers
The public Booking API uses a token bucket per API key. Free tier keys get 10 requests per second with a burst of 20. Business tier keys get 100 requests per second with a burst of 200. Enterprise limits are set per contract.

## Responses
When a key exceeds its limit the API returns HTTP 429 with a Retry-After header in seconds. Every response includes X-RateLimit-Limit and X-RateLimit-Remaining headers.

## Raising a limit
Customers can request a higher limit through their account manager. Support engineers must not change limits directly; changes go through the LIMITS ticket queue and are applied by the platform team within 2 business days.
