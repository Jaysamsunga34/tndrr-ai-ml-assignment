#!/usr/bin/env python3
"""
generate_dataset.py - Deterministic, leak-free dataset generation for Support Ticket -> Structured JSON.
Features:
  - Natural customer phrasing (no literal label leakage like 'category=billing' in text)
  - Disjoint intent descriptions across splits: descriptions 0..2 for Train/Val, description 3 exclusively for Test
  - Disjoint conversational templates across Train, Val, and Test
  - 14 hand-crafted hard edge cases (deceptive urgency, dual-intent conflicts) in Test
  - Deterministic seed=42, >= 65 held-out test examples
"""

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

INTENTS_BY_CAT = {
    "billing": [
        "refund_delay",
        "subscription_cancel",
        "payment_failed",
        "invoice_request",
        "overcharge_dispute",
        "update_payment_method"
    ],
    "technical": [
        "bug_report",
        "service_outage",
        "performance_issue",
        "api_error",
        "data_sync_failure",
        "integration_crash"
    ],
    "account": [
        "password_reset",
        "2fa_trouble",
        "profile_update",
        "account_deletion",
        "role_permission_change",
        "email_verification"
    ],
    "security": [
        "account_takeover_suspected",
        "unauthorized_access",
        "phishing_report",
        "data_breach_inquiry",
        "suspicious_ip_login"
    ],
    "general": [
        "feature_request",
        "pricing_inquiry",
        "documentation_feedback",
        "human_agent_request",
        "partnership_query"
    ]
}

# Disjoint conversational frames by urgency level (Train vs Val vs Test)
TRAIN_FRAMES = {
    "critical": [
        "EMERGENCY: {desc}. Our entire team is completely blocked right now!",
        "All hands needed immediately — {desc}. Production impact is severe.",
        "Critical incident happening now: {desc}. Please treat as P0!",
        "Stop everything, major outage/incident: {desc}.",
        "Severe emergency on our enterprise account: {desc}."
    ],
    "high": [
        "Hi support, this needs urgent attention today: {desc}.",
        "We really need this resolved quickly because {desc}.",
        "Please prioritize this request — {desc}.",
        "Important issue affecting our workflow: {desc}.",
        "Frustrated customer here, please fix soon: {desc}."
    ],
    "medium": [
        "Hello team, could you help us out? {desc}.",
        "Hi, I am reaching out because {desc}. Thanks!",
        "Can someone look into this when available? {desc}.",
        "Support request: {desc}. Appreciate your help.",
        "Good morning, having a bit of trouble — {desc}."
    ],
    "low": [
        "No rush at all, just wanted to mention: {desc}.",
        "Whenever you have a spare moment, {desc}.",
        "Quick low-priority question for your team: {desc}.",
        "Not urgent by any means, but {desc}.",
        "Just dropping a note for later: {desc}."
    ]
}

VAL_FRAMES = {
    "critical": ["Urgent P0 escalation: {desc}. Immediate response required."],
    "high": ["High-impact problem for our team today: {desc}."],
    "medium": ["Standard inquiry from our workspace: {desc}."],
    "low": ["Minor non-urgent request for whenever convenient: {desc}."]
}

TEST_FRAMES = {
    "critical": [
        "Red alert! Everything just broke — {desc}. We are losing customers every minute.",
        "Code red on our production environment: {desc}. Need immediate engineering intervention."
    ],
    "high": [
        "This is blocking our daily operations and needs prompt attention: {desc}.",
        "We've been stuck on this since morning and need a fast resolution: {desc}."
    ],
    "medium": [
        "Hey folks, hoping you can assist with something on our account: {desc}.",
        "Following up with a routine support question: {desc}."
    ],
    "low": [
        "Totally fine if this takes a few days — {desc}.",
        "Just a casual note with zero urgency: {desc}."
    ]
}

# First 3 descriptions -> Train/Val; 4th description (index 3) -> HELD-OUT TEST ONLY
INTENT_DESCRIPTIONS = {
    "refund_delay": [
        "my refund has not arrived after 14 business days",
        "still waiting on my returned purchase credit from last month",
        "the refund money promised last Tuesday still hasn't hit my bank",
        "a money-back payout approved two weeks ago never appeared on my statement"
    ],
    "subscription_cancel": [
        "stop my recurring annual renewal before it charges",
        "cancel my pro membership before the next billing cycle starts",
        "I want to terminate my paid subscription right away",
        "please end my auto-renewing seat plan and downgrade to free"
    ],
    "payment_failed": [
        "my credit card was declined during monthly checkout",
        "the subscription payment keeps failing with error code 4002",
        "autopay failed and my account banner now says past due",
        "Stripe checkout rejects our corporate debit card every time we retry"
    ],
    "invoice_request": [
        "we need the official tax invoice and receipt for fiscal year 2025",
        "please send a VAT-compliant invoice for our Q3 corporate filings",
        "the PDF download link for invoice INV-9042 is missing",
        "could you email an itemized billing receipt for our accounting audit"
    ],
    "overcharge_dispute": [
        "I was billed twice for the single-seat monthly plan",
        "there is an unrecognized $49 extra fee on my bank statement",
        "we were overcharged by 20% on the latest enterprise bill",
        "my card was charged full price even though I cancelled during the free trial"
    ],
    "update_payment_method": [
        "I want to replace my expired Mastercard with our new corporate Visa",
        "how do I switch our billing payment method from PayPal to ACH debit",
        "need help updating our primary credit card on file",
        "we want to add a backup payment card so renewals don't bounce"
    ],
    "bug_report": [
        "clicking export to CSV throws an unhandled Javascript null pointer error",
        "the settings modal freezes whenever I enter non-ASCII characters",
        "toolbar buttons vanish when toggling dark mode on iOS Safari",
        "the dashboard table pagination skips every second page when sorting"
    ],
    "service_outage": [
        "the main production API is returning 502 Bad Gateway across all regions",
        "your status page says green but the entire web dashboard is down",
        "complete platform downtime is hitting all our enterprise tenants",
        "none of our workers can connect to your hosted services globally"
    ],
    "performance_issue": [
        "search query latency has spiked to over 15 seconds per request",
        "the mobile app takes 45 seconds just to render the home feed",
        "runaway browser memory usage is causing Chrome tabs to crash",
        "scheduled analytics reports are timing out after 60 seconds consistently"
    ],
    "api_error": [
        "we keep receiving HTTP 500 internal server errors when calling /v1/ingest",
        "your REST endpoint returns malformed JSON with missing closing brackets",
        "POST /auth/token returns 403 Forbidden even with freshly rotated keys",
        "the API rate limiter triggers after 2 requests instead of our 100/sec quota"
    ],
    "data_sync_failure": [
        "our Salesforce connector has stopped syncing new contacts for 3 days",
        "outgoing webhook events are silently dropped without retries",
        "the PostgreSQL read replica is lagging 12 hours behind primary writes",
        "bidirectional sync conflicts are overwriting our local CRM records"
    ],
    "integration_crash": [
        "the Slack alert integration stopped posting after yesterday's update",
        "our Zapier webhook throws connection refused continuously",
        "the GitHub PR check runner crashes with exit status 137",
        "the Shopify store plugin disconnects automatically every 30 minutes"
    ],
    "password_reset": [
        "I forgot my master password and the recovery email never arrives",
        "the password reset token expired immediately and the link won't load",
        "locked out of my user login after 3 wrong attempts and need a reset",
        "unable to reset my forgotten account password through the login form"
    ],
    "2fa_trouble": [
        "I lost my authenticator phone and don't have backup SMS codes",
        "my 6-digit TOTP authenticator code is always rejected as invalid",
        "our YubiKey hardware token isn't recognized on the login screen",
        "can't get past two-factor verification after migrating to a new device"
    ],
    "profile_update": [
        "I cannot change our registered business name in workspace settings",
        "need to update my admin email address to our new corporate domain",
        "saving my new phone number fails the profile validation check",
        "uploading a new company avatar errors out with unsupported file format"
    ],
    "account_deletion": [
        "permanently delete my user account and purge all stored PII",
        "submitting a GDPR right-to-be-forgotten erasure request for my profile",
        "close our organization account and wipe all telemetry logs",
        "please terminate my profile and erase personal data under CCPA rules"
    ],
    "role_permission_change": [
        "promote user Sarah to organization billing administrator",
        "revoke write permissions for contractor accounts on our project",
        "custom role permissions aren't propagating down to sub-teams",
        "we need a read-only auditor role assigned to our compliance lead"
    ],
    "email_verification": [
        "the account confirmation email isn't showing up in inbox or spam",
        "clicking the email activation link says invalid cryptographic signature",
        "the resend verification email button is greyed out on signup",
        "completing email verification keeps looping back to the unverified banner"
    ],
    "account_takeover_suspected": [
        "someone changed my login email and linked an unknown device from overseas",
        "got a security alert that my password changed even though I didn't touch it",
        "an attacker has an active session in our admin console right now",
        "an unauthorized intruder is currently modifying settings inside our tenant"
    ],
    "unauthorized_access": [
        "confidential files were downloaded by a former employee's disabled account",
        "audit logs reveal API reads happening without any valid OAuth token",
        "we spotted an RBAC role bypass exposing private records on a public route",
        "internal project documents became accessible to external guests without sharing"
    ],
    "phishing_report": [
        "our staff received a fake invoice email impersonating your billing domain",
        "a spoofed lookalike domain is sending malicious links to our customers",
        "someone launched a credential phishing campaign mimicking your login portal",
        "we detected a fraudulent site harvesting logins using your brand logo"
    ],
    "data_breach_inquiry": [
        "requesting a formal disclosure on whether our tenant was hit by the recent CVE",
        "was any of our customer PII exposed in last week's third-party vendor incident",
        "our CISO needs an official security patch and exposure attestation report",
        "please confirm if our workspace data was affected by the disclosed breach"
    ],
    "suspicious_ip_login": [
        "a login alert from a Tor exit node just fired on our root admin account",
        "thousands of failed SSH/login attempts are hitting us from an unknown subnet",
        "impossible travel alert showed logins from Tokyo and London 5 minutes apart",
        "we are seeing a coordinated brute-force login spike against our portal"
    ],
    "feature_request": [
        "it would be great to have native dark mode in the desktop client",
        "please consider adding automated weekly PDF summary exports by email",
        "could you add multi-currency display support to the analytics view",
        "suggestion: let users bulk-tag multiple support threads at once"
    ],
    "pricing_inquiry": [
        "how much does the enterprise tier cost for 500 developer seats",
        "do you offer non-profit or academic discounts on annual plans",
        "what is the overage pricing structure once we exceed 1M API calls",
        "could you share a custom quote for dedicated single-tenant hosting"
    ],
    "documentation_feedback": [
        "the Python SDK quickstart snippet still uses deprecated v1 syntax",
        "your getting started guide omits two required environment variables",
        "the API reference doesn't document the X-RateLimit-Reset header format",
        "several hyperlinks in the webhook architecture docs return 404"
    ],
    "human_agent_request": [
        "your chatbot doesn't understand my issue, please connect me to a human",
        "I need to speak directly with a live support engineer or supervisor",
        "please escalate this thread to a real person who can call me back",
        "stop sending automated replies and transfer me to a senior human rep"
    ],
    "partnership_query": [
        "our consulting agency wants to join your certified reseller program",
        "reaching out regarding co-marketing and joint technology integration",
        "interested in exploring a strategic distribution partnership in Europe",
        "who is the right business development contact for vendor partnerships"
    ]
}

PRIORITY_MAP = {
    "service_outage": ["critical", "high"],
    "account_takeover_suspected": ["critical", "high"],
    "unauthorized_access": ["critical", "high"],
    "phishing_report": ["critical", "high"],
    "suspicious_ip_login": ["critical", "high"],
    "data_breach_inquiry": ["high", "critical"],
    "payment_failed": ["high", "medium"],
    "refund_delay": ["high", "medium"],
    "overcharge_dispute": ["high", "medium"],
    "bug_report": ["high", "medium", "low"],
    "api_error": ["critical", "high", "medium"],
    "data_sync_failure": ["high", "medium"],
    "integration_crash": ["high", "medium"],
    "performance_issue": ["high", "medium"],
    "password_reset": ["high", "medium"],
    "2fa_trouble": ["high", "medium"],
    "account_deletion": ["high", "medium"],
    "role_permission_change": ["medium", "low"],
    "email_verification": ["medium", "low"],
    "profile_update": ["medium", "low"],
    "subscription_cancel": ["high", "medium"],
    "invoice_request": ["medium", "low"],
    "update_payment_method": ["medium", "low"],
    "human_agent_request": ["high", "medium"],
    "feature_request": ["low"],
    "pricing_inquiry": ["low", "medium"],
    "documentation_feedback": ["low"],
    "partnership_query": ["low", "medium"]
}

# 14 Challenging real-world edge cases exclusively in held-out test set
HARD_EDGE_CASES = [
    {
        "input": "URGENT PLEASE READ: Actually I'm not in a hurry at all, whenever you have time next week could you share enterprise pricing for 50 seats?",
        "expected": {"category": "general", "priority": "low", "intent": "pricing_inquiry"}
    },
    {
        "input": "Critical security alert!! Wait nevermind, false alarm—my coworker John just forgot his login password again. Can we trigger a password reset for him?",
        "expected": {"category": "account", "priority": "medium", "intent": "password_reset"}
    },
    {
        "input": "Our entire production database cluster went unreachable right after your v4.3 rollout! Every customer transaction is failing right now!",
        "expected": {"category": "technical", "priority": "critical", "intent": "service_outage"}
    },
    {
        "input": "An unknown user from a foreign IP is actively exporting our entire customer table from the admin panel as I type this!",
        "expected": {"category": "security", "priority": "critical", "intent": "account_takeover_suspected"}
    },
    {
        "input": "Hey team, not urgent at all—just thought it would be cool if you added a button to export monthly invoices as CSV instead of only PDF.",
        "expected": {"category": "general", "priority": "low", "intent": "feature_request"}
    },
    {
        "input": "I was double-charged $299 on my credit card statement because your checkout API threw a 500 error while processing my renewal.",
        "expected": {"category": "billing", "priority": "high", "intent": "overcharge_dispute"}
    },
    {
        "input": "I'm locked out because clicking the password reset endpoint returns a 502 Bad Gateway server crash every time.",
        "expected": {"category": "technical", "priority": "high", "intent": "api_error"}
    },
    {
        "input": "Zero rush! Just spotted a small typo in the webhook authentication code example on page 3 of your developer docs.",
        "expected": {"category": "general", "priority": "low", "intent": "documentation_feedback"}
    },
    {
        "input": "I've been arguing with your AI bot for 45 minutes and getting nowhere. Escalate me to a real human supervisor immediately!",
        "expected": {"category": "general", "priority": "high", "intent": "human_agent_request"}
    },
    {
        "input": "Per GDPR Article 17 legal mandate, we must permanently delete and purge all personal data for user ID #88391 within 24 hours.",
        "expected": {"category": "account", "priority": "high", "intent": "account_deletion"}
    },
    {
        "input": "My CFO is furious—we canceled our annual subscription in October yet you billed our corporate Visa again this morning!",
        "expected": {"category": "billing", "priority": "high", "intent": "overcharge_dispute"}
    },
    {
        "input": "Please don't panic, just checking casually whether our workspace was impacted at all by last month's supply chain CVE.",
        "expected": {"category": "security", "priority": "high", "intent": "data_breach_inquiry"}
    },
    {
        "input": "My phone fell in a lake so my Google Authenticator is gone and I have an investor demo in 20 minutes—help me bypass 2FA!",
        "expected": {"category": "account", "priority": "high", "intent": "2fa_trouble"}
    },
    {
        "input": "Could you downgrade Sarah's role from admin to read-only before she switches teams next Friday? No hurry.",
        "expected": {"category": "account", "priority": "low", "intent": "role_permission_change"}
    }
]

def build_datasets():
    train_data, val_data, test_data = [], [], []

    for cat, intents in INTENTS_BY_CAT.items():
        for intent in intents:
            descs = INTENT_DESCRIPTIONS[intent]
            train_descs = descs[:3]
            test_desc = descs[3]
            prios = PRIORITY_MAP[intent]

            # Train: combine train_descs x prios x subset of TRAIN_FRAMES
            for desc in train_descs:
                for prio in prios:
                    for frame in TRAIN_FRAMES[prio][:3]:
                        train_data.append({
                            "input": frame.format(desc=desc),
                            "expected": {"category": cat, "priority": prio, "intent": intent}
                        })

            # Val: 1 sample per intent with VAL_FRAMES
            v_prio = prios[0]
            v_desc = train_descs[0]
            val_data.append({
                "input": VAL_FRAMES[v_prio][0].format(desc=v_desc),
                "expected": {"category": cat, "priority": v_prio, "intent": intent}
            })

            # Test: 2 samples per intent using UNSEEN test_desc and UNSEEN TEST_FRAMES
            for idx_p, t_prio in enumerate(prios[:2]):
                frame = TEST_FRAMES[t_prio][idx_p % len(TEST_FRAMES[t_prio])]
                test_data.append({
                    "input": frame.format(desc=test_desc),
                    "expected": {"category": cat, "priority": t_prio, "intent": intent}
                })

    # Add 14 hard edge cases to test set
    test_data.extend(HARD_EDGE_CASES)

    random.seed(SEED)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Strict de-duplication and zero-overlap verification
    train_inputs = {x["input"] for x in train_data}
    seen_test = set()
    clean_test = []
    for x in test_data:
        if x["input"] not in train_inputs and x["input"] not in seen_test:
            seen_test.add(x["input"])
            clean_test.append(x)

    return train_data, val_data, clean_test

def main():
    base_dir = Path("/Users/jayadityashah/Downloads/tndrr-ai-ml-assignment/task1_finetuning/data")
    base_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = build_datasets()
    print(f"Leak-free Dataset generated (seed={SEED}):")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples (including {len(HARD_EDGE_CASES)} hard edge cases; req >= 50)")
    assert len(test) >= 50

    for name, ds in [("train.json", train), ("val.json", val), ("test.json", test)]:
        with open(base_dir / name, "w", encoding="utf-8") as f:
            json.dump(ds, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
