# Software Release Process
Releases ship on a weekly train that cuts every Tuesday at 16:00 UTC. Code must be merged to
main with two approving reviews and a green CI pipeline, including unit, integration, and
security scans. Feature flags gate all user-visible changes; a flag must be able to roll back a
feature within minutes without a deploy. Canary deployment routes 5 percent of traffic for at
least 2 hours before full rollout, and error-budget burn above 2 percent halts promotion
automatically. Hotfixes outside the train require approval from the release manager and an
attached incident or ticket. Database migrations must be backward compatible one version.
