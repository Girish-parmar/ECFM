-- Week 11 (S19): PostgreSQL + TimescaleDB schema for the platform's system of record.
-- Run in the course database container:  psql -f schema.sql
-- Exercise: add the `orders` table (see the README) and the indexes your queries need.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE instruments (
  instrument_id serial PRIMARY KEY,
  symbol        text NOT NULL,
  asset_class   text NOT NULL CHECK (asset_class IN ('EQUITY','FUTURE','OPTION','FX','CRYPTO')),
  currency      char(3) NOT NULL DEFAULT 'USD',
  tick          numeric NOT NULL CHECK (tick > 0),
  multiplier    numeric NOT NULL DEFAULT 1 CHECK (multiplier > 0),
  UNIQUE (symbol, asset_class)
);

CREATE TABLE bars (
  instrument_id int NOT NULL REFERENCES instruments,
  timeframe     text NOT NULL,                 -- '1m', '5m', '1d'
  ts            timestamptz NOT NULL,          -- bar OPEN time, UTC
  open numeric, high numeric, low numeric, close numeric, volume numeric,
  source        text NOT NULL,
  PRIMARY KEY (instrument_id, timeframe, ts),
  CHECK (low <= LEAST(open, close) AND high >= GREATEST(open, close))
);
SELECT create_hypertable('bars', 'ts');

CREATE TABLE fills (
  fill_id         text PRIMARY KEY,            -- broker execution id: replays cannot double-count
  client_order_id text NOT NULL,
  instrument_id   int NOT NULL REFERENCES instruments,
  ts              timestamptz NOT NULL,
  qty             numeric NOT NULL,            -- + buy, - sell
  price           numeric NOT NULL,
  fee             numeric NOT NULL DEFAULT 0
);

-- Daily realized cash flow per instrument (window functions exercise: turn this into running P&L)
-- SELECT instrument_id, date_trunc('day', ts) AS day, sum(-qty * price - fee) AS cash
-- FROM fills GROUP BY 1, 2 ORDER BY 1, 2;
