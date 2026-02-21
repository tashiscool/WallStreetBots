# ADR-002: Enforce PR-only delivery workflow
## Status: accepted
## Context: Autonomous changes must be auditable and reversible.
## Decision: All automated code changes must land via pull requests; direct push to main is disallowed.
## Consequences: Slightly slower cycle time with materially better reviewability.
## Alternatives Considered: Direct push automation; rejected due to operational risk.
