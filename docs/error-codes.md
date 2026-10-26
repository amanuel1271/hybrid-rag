# Booking API Error Codes

## SHP-4031: Address validation failed
Returned when the destination address cannot be geocoded. The response includes the field that failed. Clients should ask the user to correct the address; retrying the same request will not help.

## SHP-4090: Duplicate booking
Returned when a booking with the same Idempotency-Key was already created. The response body contains the original booking id, so clients can treat it as a success.

## SHP-5003: Carrier timeout
Returned when the upstream carrier did not answer within 8 seconds. This error is safe to retry with exponential backoff, starting at 1 second and capped at 30 seconds.

## SHP-5020: Pricing engine unavailable
Returned when the pricing service is down. Bookings cannot be priced, so the API rejects them; quotes are not cached for more than 60 seconds.
