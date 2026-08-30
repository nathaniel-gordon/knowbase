"""The bundled demo knowledge base and its labelled evaluation suites.

This is fixture data, so it lives in the app layer rather than the domain: the domain has no
opinion about which documents exist. Point ``/ingest`` at your own folder and the same
use-cases run unchanged.

Two suites are provided:

``core``
    12 questions that share vocabulary with the source documents, which is the realistic case
    for a KB search box. Every index adapter saturates this suite; run ``python run.py demo``
    to see the current numbers.
``hard``
    12 paraphrase questions written to avoid the documents' own wording ("thumb drive" for
    "removable USB storage"). Lexical retrieval is genuinely stressed here, which is what makes
    the ``/compare`` A/B endpoint informative instead of a three-way tie at 1.0.
"""
from __future__ import annotations

from pathlib import Path

from ..domain.model import EvalCase

DOCS: dict[str, str] = {
    "it-security-policy": """# IT Security Policy
All employees must enable multi-factor authentication (MFA) on their corporate accounts within
48 hours of onboarding. Passwords must be at least 14 characters and rotated every 180 days.
Laptops must run full-disk encryption (BitLocker on Windows, FileVault on macOS) and the
endpoint agent at all times. Removable USB storage is prohibited for confidential data.
Suspected phishing emails must be reported with the Report Phish button within one hour of
receipt. Security exceptions require written approval from the CISO office and expire after
90 days. Production credentials must never be stored in source code, tickets, or chat; use the
central secrets vault. Quarterly security-awareness training is mandatory and tracked.""",

    "vpn-remote-access": """# VPN and Remote Access Guide
Remote access to internal systems requires the GlobalConnect VPN client. Install it from the
self-service portal, sign in with SSO, and approve the MFA push. Split tunneling is disabled by
policy; all traffic routes through the corporate gateway. If the VPN disconnects repeatedly,
first switch from UDP to TCP mode in Settings, then verify your home router's MTU is at least
1400, and finally reinstall the client. Contractors receive time-boxed VPN certificates valid
for 30 days, renewable by their sponsoring manager. Access to the production subnet additionally
requires membership in the prod-access group, granted through a Tier-2 request with director
approval.""",

    "employee-onboarding": """# Employee Onboarding Checklist
Day 1: collect your laptop from IT, sign the acceptable-use agreement, enable MFA, and join the
#new-joiners channel. Week 1: complete security-awareness training, set up the VPN, meet your
onboarding buddy, and read your team's runbook. Your manager schedules a 30/60/90-day plan in
the first week. Payroll enrollment closes on the 20th of each month; submissions after that are
paid the following cycle. Benefits elections must be completed within 30 days of your start
date. New engineers should request repository access through the code-access workflow and
complete the secure-coding module before their first deploy.""",

    "expense-policy": """# Travel and Expense Policy
Expenses must be submitted within 30 days of being incurred; reports older than 90 days are
rejected automatically. Daily meal allowance is 65 USD domestic and 85 USD international, with
itemized receipts required for any single expense over 25 USD. Economy class is required for
flights under 6 hours; premium economy is allowed for longer flights with VP approval. Hotel
bookings must go through the corporate travel portal unless the portal rate exceeds the city
cap by 20 percent. Client entertainment above 150 USD per head needs pre-approval from the
account executive's director. Personal mileage is reimbursed at the prevailing IRS rate.""",

    "incident-runbook": """# Production Incident Runbook
When an alert fires, the on-call engineer acknowledges within 5 minutes and opens an incident
channel named inc-<date>-<slug>. Severity levels: SEV1 means customer-facing outage, SEV2 means
degraded service with workaround, SEV3 means internal-only impact. SEV1 requires paging the
incident commander and posting status updates every 30 minutes. Mitigation always precedes root
cause: prefer rollback over forward-fix during business hours. After resolution, a blameless
postmortem is due within 5 business days, with action items tracked to completion. Never test
fixes directly in production; use the staging mirror with traffic replay.""",

    "release-process": """# Software Release Process
Releases ship on a weekly train that cuts every Tuesday at 16:00 UTC. Code must be merged to
main with two approving reviews and a green CI pipeline, including unit, integration, and
security scans. Feature flags gate all user-visible changes; a flag must be able to roll back a
feature within minutes without a deploy. Canary deployment routes 5 percent of traffic for at
least 2 hours before full rollout, and error-budget burn above 2 percent halts promotion
automatically. Hotfixes outside the train require approval from the release manager and an
attached incident or ticket. Database migrations must be backward compatible one version.""",

    "data-retention": """# Data Retention and Privacy Standard
Customer personal data is retained for the life of the contract plus 24 months, then purged by
the quarterly deletion job. Application logs are kept 90 days in hot storage and 13 months in
cold archive. Backups follow a 3-2-1 scheme and expire after 35 days. Data subject deletion
requests must be honored within 30 days and cascaded to sub-processors within 7 days of intake.
Production data may never be copied to laptops or non-production environments; use the masked
synthetic dataset for development. Data residency: EU customer data stays in the eu-central
region, including backups and analytics derivatives.""",

    "pto-policy": """# Paid Time Off Policy
Full-time employees accrue 1.75 PTO days per month, or 21 days per year, in addition to public
holidays. Up to 5 unused days roll over to the next calendar year and expire on March 31.
Requests longer than 10 consecutive working days need approval one month in advance. Sick leave
is separate and unlimited with a doctor's note required from the fourth consecutive day.
Parental leave is 16 weeks fully paid for primary caregivers and 6 weeks for secondary
caregivers, usable within 12 months of birth or adoption. On-call engineers receive one
compensation day per weekend on rotation, to be used within 60 days.""",

    "code-review-guidelines": """# Code Review Guidelines
Every change requires two approvals, one of which must come from a code owner of the touched
directory. Reviews should complete within one business day; authors may escalate stale reviews
in the team channel after 24 hours. Keep pull requests under 400 changed lines where practical;
larger changes should be split or accompanied by a design doc. Reviewers check correctness,
tests, security implications, and observability, not style, which is enforced by linters.
Nit-level comments must be prefixed "nit:" and are non-blocking. Self-merges are prohibited
except for documentation-only changes with a green pipeline.""",

    "product-faq": """# Atlas Platform - Customer FAQ
Atlas supports SSO via SAML 2.0 and OIDC on the Enterprise plan; SCIM provisioning syncs users
hourly. The API rate limit is 600 requests per minute per workspace, with burst credits of 2x
for 60 seconds; exceeding it returns HTTP 429 with a Retry-After header. Data export is
available in CSV and Parquet, generated asynchronously and delivered to a signed URL valid for
24 hours. Uptime SLA is 99.9 percent monthly for Enterprise, with 10 percent service credit per
0.1 percent shortfall. Webhooks retry with exponential backoff for 24 hours; configure a dead-
letter endpoint to capture failures beyond that window.""",

    "oncall-handbook": """# On-Call Handbook
Rotations run Monday 10:00 to Monday 10:00 with a primary and a shadow. The primary must be
reachable within 5 minutes and have a laptop with VPN access at hand; the shadow takes over
automatically if two pages go unacknowledged. Alert fatigue is managed by a weekly triage
review; any alert that fired more than 3 times without action must be tuned or deleted. Silences
longer than 24 hours require a linked ticket. Handoff notes are mandatory: open incidents,
recent deploys, and known risks. On-call compensation follows the PTO policy's rotation-day
rule.""",

    "procurement-guide": """# Vendor and Procurement Guide
New software vendors require a security review (SIG-Lite questionnaire) and a data-processing
agreement before any data is shared. Purchases under 5,000 USD annually can be approved by a
budget owner; 5,000 to 50,000 USD requires finance review; above 50,000 USD goes to the
procurement committee, which meets bi-weekly. Contracts auto-renewing must be flagged 60 days
before the renewal date in the vendor register. Preferred payment terms are net-45. Any vendor
processing customer personal data must be added to the public sub-processor list at least 30
days before go-live, allowing customers to object.""",

    "ml-platform-guide": """# Internal ML Platform Guide
Model training jobs run on the Kubeflow cluster; request quota through the ml-infra portal.
Every production model must be registered in the model registry with a model card covering
intended use, training data lineage, evaluation metrics, and known limitations. Models serving
customer traffic require shadow evaluation for one week and sign-off from the ML review board.
Feature pipelines are defined in the feature store; ad-hoc features computed in notebooks may
not ship to production. Drift monitoring alerts when population stability index exceeds 0.2,
triggering scheduled retraining. GPU quota resets monthly and unused quota does not roll over.""",

    "brand-communications": """# Brand and External Communications Policy
Only authorized spokespeople may make statements to press or analysts; route inquiries to the
communications team within 2 hours. Customer names or logos may not be used in marketing
material without a signed reference agreement. Social posts about unreleased features are
prohibited until the embargo lifts. The company blog follows a two-reviewer rule: one editorial,
one legal for posts touching customers, partners, security, or financials. Crisis communications
follow the incident process: communications lead joins any SEV1 with external visibility and
owns the public status page updates.""",
}

CORE_SUITE: tuple[EvalCase, ...] = tuple(EvalCase(q, d) for q, d in [
    ("How long do I have to submit an expense report?", "expense-policy"),
    ("What is the API rate limit per workspace?", "product-faq"),
    ("When does the weekly release train cut?", "release-process"),
    ("How many PTO days do employees get per year?", "pto-policy"),
    ("What should I do if the VPN keeps disconnecting?", "vpn-remote-access"),
    ("How quickly must a data subject deletion request be honored?", "data-retention"),
    ("What severity is a customer-facing outage and how often are status updates posted?",
     "incident-runbook"),
    ("How many approvals does a pull request need?", "code-review-guidelines"),
    ("What password length is required by the security policy?", "it-security-policy"),
    ("When does drift monitoring trigger retraining on the ML platform?", "ml-platform-guide"),
    ("Who can approve a software purchase under 5000 USD?", "procurement-guide"),
    ("How fast must the on-call primary acknowledge a page?", "oncall-handbook"),
])

HARD_SUITE: tuple[EvalCase, ...] = tuple(EvalCase(q, d) for q, d in [
    ("Can I copy sensitive files onto a thumb drive?", "it-security-policy"),
    ("My connection to the office network keeps dropping, what should I try?",
     "vpn-remote-access"),
    ("A journalist emailed me for comment, what now?", "brand-communications"),
    ("Is there a size limit on the changes I send for review?", "code-review-guidelines"),
    ("What happens to holiday I did not use by December?", "pto-policy"),
    ("How long before old application logs move out of hot storage?", "data-retention"),
    ("Do I need sign-off before my model serves live customer traffic?", "ml-platform-guide"),
    ("What are the standard payment terms for suppliers?", "procurement-guide"),
    ("Can I book a better seat for a long-haul flight?", "expense-policy"),
    ("Nobody answered the page, who picks it up?", "oncall-handbook"),
    ("What do I need to finish on my very first day?", "employee-onboarding"),
    ("What happens if we send too many API calls?", "product-faq"),
])

SUITES: dict[str, tuple[EvalCase, ...]] = {"core": CORE_SUITE, "hard": HARD_SUITE}


def write_demo_corpus(destination: Path) -> int:
    """Materialise the bundled KB as .md files so the filesystem adapter has real input."""
    destination.mkdir(parents=True, exist_ok=True)
    for doc_id, text in DOCS.items():
        (destination / f"{doc_id}.md").write_text(text.strip() + "\n", encoding="utf-8")
    return len(DOCS)
