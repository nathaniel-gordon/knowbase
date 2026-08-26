"""The tiny corpus every contract battery is written against.

It lives with the batteries rather than in ``conftest`` because the contracts own it: the
labelled probes below are part of the IndexPort promise ("a labelled question retrieves its
document at rank 1"), so they must travel with the contract, not with the test session.
"""
from __future__ import annotations

FIXTURE_DOCS: dict[str, str] = {
    "expenses": "# Expense Rules\n"
                "Expense reports must be submitted within 30 days of the purchase date. "
                "Receipts are required for any single item above 25 USD.",
    "leave": "# Time Off Rules\n"
             "Employees receive 21 paid vacation days each year and accrue them monthly. "
             "Unused vacation expires on March 31 of the following year.",
    "passwords": "# Password Rules\n"
                 "Passwords must be at least 14 characters long and rotated every 180 days. "
                 "Reused passwords are rejected by the identity provider.",
    "vpn": "# Remote Access Rules\n"
           "The GlobalConnect VPN client is required for remote access to internal systems. "
           "Set the router MTU to 1400 if the tunnel drops repeatedly.",
}

# (question, document expected at rank 1) -- the IndexPort quality floor.
FIXTURE_PROBES: tuple[tuple[str, str], ...] = (
    ("How many vacation days do employees receive each year?", "leave"),
    ("What password length is required?", "passwords"),
    ("How long do I have to submit an expense report?", "expenses"),
    ("Which VPN client is required for remote access?", "vpn"),
)
