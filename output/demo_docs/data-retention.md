# Data Retention and Privacy Standard
Customer personal data is retained for the life of the contract plus 24 months, then purged by
the quarterly deletion job. Application logs are kept 90 days in hot storage and 13 months in
cold archive. Backups follow a 3-2-1 scheme and expire after 35 days. Data subject deletion
requests must be honored within 30 days and cascaded to sub-processors within 7 days of intake.
Production data may never be copied to laptops or non-production environments; use the masked
synthetic dataset for development. Data residency: EU customer data stays in the eu-central
region, including backups and analytics derivatives.
