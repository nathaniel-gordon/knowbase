# Production Incident Runbook
When an alert fires, the on-call engineer acknowledges within 5 minutes and opens an incident
channel named inc-<date>-<slug>. Severity levels: SEV1 means customer-facing outage, SEV2 means
degraded service with workaround, SEV3 means internal-only impact. SEV1 requires paging the
incident commander and posting status updates every 30 minutes. Mitigation always precedes root
cause: prefer rollback over forward-fix during business hours. After resolution, a blameless
postmortem is due within 5 business days, with action items tracked to completion. Never test
fixes directly in production; use the staging mirror with traffic replay.
