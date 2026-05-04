"""Audit log primitives.

Returns a normalized event dict the caller persists to its own audit
table. fitapp-core is DB-agnostic on purpose — Vitalstack's enterprise
audit log has different retention + replication requirements than
CoachHQ's SaaS audit log, and FitApp's consumer audit log is a third
shape. All three call audit_event() to construct the row, then write
it themselves.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypedDict


class AuditAction(str, Enum):
    # PII / PHI reads
    READ_PROFILE = "read.profile"
    READ_MEAL = "read.meal"
    READ_LAB = "read.lab"
    READ_GLUCOSE = "read.glucose"
    READ_CYCLE = "read.cycle"
    READ_ALLERGY = "read.allergy"
    READ_MEDICATION = "read.medication"
    EXPORT_PHI = "export.phi"

    # Writes
    CREATE_USER = "create.user"
    UPDATE_PROFILE = "update.profile"
    LOG_MEAL = "create.meal"
    DELETE_MEAL = "delete.meal"
    SAVE_LAB = "create.lab"
    SAVE_ALLERGY = "create.allergy"
    SAVE_CONDITION = "create.condition"
    SAVE_MEDICATION = "create.medication"

    # Auth + admin
    LOGIN = "auth.login"
    LOGIN_FAILED = "auth.login.failed"
    LOGOUT = "auth.logout"
    PASSWORD_RESET = "auth.password.reset"
    ADMIN_IMPERSONATE = "admin.impersonate"
    ADMIN_PROVISION_TENANT = "admin.tenant.create"
    ADMIN_GRANT_ROLE = "admin.role.grant"

    # Subscription / billing
    SUBSCRIBE = "billing.subscribe"
    CANCEL = "billing.cancel"
    UPGRADE = "billing.upgrade"


class AuditEvent(TypedDict):
    timestamp: str
    actor_id: str | None
    actor_type: str
    action: str
    resource_type: str
    resource_id: str | None
    tenant_id: str | None
    ip_hash: str | None
    user_agent: str | None
    details_json: dict[str, Any]
    digest: str  # SHA-256 over the event for tamper-evident chain


def _hash_ip(ip: str | None, salt: str = "") -> str | None:
    """Hash IP with a server-side salt. Salt should rotate quarterly per
    HIPAA guidance — caller decides salt strategy."""
    if not ip:
        return None
    h = hashlib.sha256()
    h.update(salt.encode())
    h.update(ip.encode())
    return h.hexdigest()[:32]


def audit_event(
    *,
    actor_id: str | None,
    actor_type: str = "user",
    action: AuditAction | str,
    resource_type: str,
    resource_id: str | None = None,
    tenant_id: str | None = None,
    ip: str | None = None,
    ip_salt: str = "",
    user_agent: str | None = None,
    details: dict[str, Any] | None = None,
    prev_digest: str | None = None,
) -> AuditEvent:
    """Construct a normalized audit event row.

    `prev_digest`: pass the digest of the previous event in this chain
    to build a tamper-evident hash chain (Vitalstack-style PHI audit).
    Pass None for non-chained logs.
    """
    ts = datetime.now(timezone.utc).isoformat()
    action_str = action.value if isinstance(action, AuditAction) else action
    payload = details or {}

    # Compute event digest over a canonical-ordered JSON body. Including
    # prev_digest in the input makes the chain tamper-evident.
    body = {
        "timestamp": ts,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "action": action_str,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "tenant_id": tenant_id,
        "details": payload,
        "prev_digest": prev_digest,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()

    return {
        "timestamp": ts,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "action": action_str,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "tenant_id": tenant_id,
        "ip_hash": _hash_ip(ip, ip_salt),
        "user_agent": (user_agent or "")[:500] or None,
        "details_json": payload,
        "digest": digest,
    }
