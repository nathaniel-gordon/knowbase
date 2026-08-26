# Atlas Platform - Customer FAQ
Atlas supports SSO via SAML 2.0 and OIDC on the Enterprise plan; SCIM provisioning syncs users
hourly. The API rate limit is 600 requests per minute per workspace, with burst credits of 2x
for 60 seconds; exceeding it returns HTTP 429 with a Retry-After header. Data export is
available in CSV and Parquet, generated asynchronously and delivered to a signed URL valid for
24 hours. Uptime SLA is 99.9 percent monthly for Enterprise, with 10 percent service credit per
0.1 percent shortfall. Webhooks retry with exponential backoff for 24 hours; configure a dead-
letter endpoint to capture failures beyond that window.
