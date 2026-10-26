# Deploy Process

## Release trains
Production deploys happen through release trains at 11:00 and 16:00 KST on weekdays. There are no deploys on Fridays after 14:00 and none on public holidays, except hotfixes.

## Hotfixes
A hotfix needs approval from the on-call primary and one senior engineer. Hotfix branches are named hotfix/<ticket-id> and must include a regression test.

## Canary
Every deploy first goes to the canary pool, which receives 5% of traffic for 20 minutes. The deploy is rolled back automatically if the error rate on the canary exceeds 1% or p99 latency rises by more than 30% compared with the stable pool.

## Rollback
Engineers can roll back manually with `orbit deploy rollback --to <release-id>`. A rollback takes about 3 minutes and does not require approval.
