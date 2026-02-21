# Model Module Deprecation Plan

## Goal
Move all canonical Django model ownership to:

- `backend/tradingbot/models/models.py`
- `backend/tradingbot/models/__init__.py`

and fully retire legacy drift from:

- `backend/tradingbot/models.py`

## Why
The codebase historically carried two model definitions with overlapping names.
That creates schema drift risk, inconsistent behavior, and unclear ownership for
financial records.

## Current Decision (Implemented)

1. Canonical transaction ledger model is now `TradeTransaction` in
   `backend/tradingbot/models/models.py`.
2. `stock_trade` writes to `TradeTransaction` for every submitted trade side.
3. Legacy `StockTrade` in `backend/tradingbot/models.py` is now a compatibility
   alias to canonical `TradeTransaction`.
4. No new model development is allowed in `backend/tradingbot/models.py`.

## Deprecation Phases

### Phase 1: Freeze (complete in this change)
- Freeze legacy module.
- Introduce canonical transaction model.
- Route live write path to canonical model.

### Phase 2: Consumer Cutover (target: March 20, 2026)
- Audit imports and serializers that rely on legacy structures.
- Update any remaining callers to import from `backend.tradingbot.models`.
- Validate reporting and analytics read paths against `TradeTransaction`.

### Phase 3: Legacy Removal (target: April 24, 2026)
- Remove `backend/tradingbot/models.py` from the repository.
- Remove compatibility alias `StockTrade`.
- Keep migration history intact; do not rewrite historical migrations.

## Enforcement Rules

- All new models and model edits must happen in `backend/tradingbot/models/models.py`.
- All new imports must use package path `backend.tradingbot.models`.
- Legacy file is read-only compatibility surface until Phase 3 removal.

## Acceptance Criteria

- No production code reads/writes legacy-only model shapes.
- Canonical transaction history is represented by `TradeTransaction`.
- CI passes (`ruff`, `pytest`) with legacy compatibility intact.
