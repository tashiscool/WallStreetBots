-- Tax lots and realized PnL tracking for wash sale compliance
-- PostgreSQL schema

CREATE TABLE IF NOT EXISTS tax_lots (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  open_ts TIMESTAMPTZ NOT NULL,
  qty NUMERIC(18,6) NOT NULL CHECK (qty > 0),
  cost NUMERIC(18,6) NOT NULL,
  remaining NUMERIC(18,6) NOT NULL,
  closed_ts TIMESTAMPTZ,
  method TEXT NOT NULL DEFAULT 'FIFO'
);

CREATE TABLE IF NOT EXISTS realizations (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  qty NUMERIC(18,6) NOT NULL,
  proceed NUMERIC(18,6) NOT NULL,
  cost NUMERIC(18,6) NOT NULL,
  realized_pnl NUMERIC(18,6) NOT NULL,
  wash_disallowed NUMERIC(18,6) NOT NULL DEFAULT 0,
  open_refs JSONB,
  close_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol ON tax_lots(symbol);
CREATE INDEX IF NOT EXISTS idx_tax_lots_open_ts ON tax_lots(open_ts);
CREATE INDEX IF NOT EXISTS idx_realizations_symbol ON realizations(symbol);
CREATE INDEX IF NOT EXISTS idx_realizations_close_ts ON realizations(close_ts);