import numpy as np
import pytest
from _loader import load
from common import sector_universe

bk = load("clinic_w2_book", "book")
PRICES, SECTORS, TRUTH = sector_universe(n_sectors=6, per_sector=8, n_days=1000, pairs_per_sector=3, seed=7)


@pytest.fixture(scope="module")
def book():
    return bk.build_book(PRICES, SECTORS)


def test_market_betas():
    b = bk.market_betas(PRICES)
    assert list(b.index) == list(PRICES.columns)
    assert b.mean() == pytest.approx(1.0)                                           # the average stock IS the market
    r = PRICES.iloc[:750].diff().dropna()
    m = r.mean(axis=1)
    assert b["S0_0"] == pytest.approx(np.cov(r["S0_0"], m)[0, 1] / m.var())


def test_build_book(book):
    assert len(book["pairs"]) >= 15 and set(TRUTH) <= set(book["pairs"].values())
    assert book["weights"].sum() == pytest.approx(1) and (book["weights"] > 0).all()
    assert book["returns"].index[0] == PRICES.index[750] and len(book["returns"]) == 250
    np.testing.assert_allclose(book["returns"], book["pnl"].to_numpy() @ book["weights"][book["pnl"].columns])
    ex = book["exposures"]
    assert list(ex.columns) == ["gross", "net", "net_beta"]
    assert ex["gross"].max() <= 1 + 1e-9 and ex["net_beta"].abs().max() < 0.05      # market neutral


def test_risk_report(book):
    rep = bk.risk_report(book)
    assert rep["n_pairs"] == len(book["pairs"]) and rep["max_name_load"] <= 0.15 + 1e-9
    assert rep["sharpe"] > 1 and rep["max_drawdown"] <= 0 and rep["var_99"] > 0
    assert rep["crowded_unwind"] == pytest.approx(3 * (book["weights"] * book["pnl"].std()).sum())
    assert rep["unwind_vs_var"] > 3                                                  # diversification is not a hedge


def test_tight_name_cap():
    tight = bk.build_book(PRICES, SECTORS, name_cap=0.06)
    assert bk.risk_report(tight)["max_name_load"] <= 0.06 + 1e-9
