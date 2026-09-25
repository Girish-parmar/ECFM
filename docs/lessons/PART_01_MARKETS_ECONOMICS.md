# Part 1 — Financial Markets & Economic Foundations: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 1, **Month 1** (program weeks 1–4) |
| Format | 16 sessions × 90 min (4 per week) + 4 lab clinics × 120 min + self-study (~6 hrs/week) |
| Total effort | ~24 hrs live + 8 hrs clinic + 24 hrs self-study ≈ **56 hours** |
| Tools | **Excel** (main lab tool this month), ready-to-run Jupyter notebooks (learners run and modify them; writing Python from scratch starts in Part 3), IB and Alpaca **paper** accounts (read-only exploration of quotes, chains, contract specs) |
| Data | FRED / ALFRED (macro series and their historical vintages), SEC EDGAR, CME contract specifications, broker quotes and option chains, sample tick/quote files provided by the course |
| Graded deliverables | Macro + microstructure research report · Excel workbook (returns, futures fair value and margin ledger, option payoffs, Black–Scholes price and Greeks) · end-of-month quiz |
| Covers original items | "Stock Market Basic" 1 (macroeconomics), 7 (financial markets), 8 (microstructure), 9 (stock, index, forex), 10 (F&O basics), 11 (stock market basics), 12 (Excel primer), 13 (derivatives), 14 (F&O strategies), 16 (algorithmic trading, orientation) |

**Why this Part matters for an algorithmic trader:** every later model assumes something about how markets work: when data is released and when it is revised, how an order is matched, what the spread costs, how a futures contract rolls, what an option's price depends on. Mistakes here become silent bugs in backtests later (look-ahead from revised data, ignored corporate actions, wrong multipliers, unrealistic fills). This month builds that ground truth, with US markets as the reference (IB and Alpaca) and no SEBI-specific content.

---

## 1. Learning Objectives

By the end of Part 1 the learner will be able to:

1. **Describe** the algorithmic trading landscape: participants, strategy families, infrastructure, and where this program's platform fits.
2. **Interpret** major macroeconomic releases (growth, inflation, employment), central-bank policy and the yield curve, and explain how they move equities, rates, the dollar and commodities, including **data revisions and point-in-time data**.
3. **Derive** market-implied policy expectations from fed funds futures and build a macro regime dashboard.
4. **Explain** how equities, ETFs, indices, FX, bonds, futures, options and crypto are structured, traded, cleared and settled; compute index levels, ETF premiums, margin and short-sale costs.
5. **Apply** stock-market mechanics: order types, sessions, auctions, corporate actions and price adjustment, financial statements and basic valuation.
6. **Analyze** market microstructure: order books, spreads and their components, fragmentation, payment for order flow, market impact, volatility controls; estimate spreads from data.
7. **Price and reason about** futures (cost of carry, basis, rolls, margin) and options (payoffs, put-call parity, no-arbitrage bounds, the inputs that drive value).
8. **Construct and analyze** standard futures and options strategies by payoff, risk and market view.
9. **Build** clean, auditable Excel models for returns, risk, futures and option pricing, and export data for the Python work that starts in Part 3.

---

## 2. Prerequisites & Setup

| Requirement | Detail |
|---|---|
| Knowledge | Comfort with algebra, logarithms/exponentials, basic probability (mean, variance, normal distribution). No programming required this month. |
| Software | Excel (Microsoft 365 recommended for `XLOOKUP`, dynamic arrays and `LET`), JupyterLab (course image), a PDF reader for filings |
| Accounts | IB paper account and Alpaca paper account opened in week 1 (used again from Part 4); free FRED account (API key for the provided notebooks) |
| Habit introduced | **Research log** from day 1: every analysis records data source, download date, data vintage and assumptions (used heavily from Part 8) |

---

## 3. Weekly Overview

| Week | Theme | Sessions | Clinic Lab | Output |
|---|---|---|---|---|
| **W1** | Orientation & macroeconomics | S1 The algorithmic trading landscape · S2 Macro I: growth, inflation, jobs & data releases · S3 Macro II: central banks, rates & the yield curve · S4 Macro III: business cycle, the dollar, commodities & intermarket | Macro regime dashboard (Excel + provided notebook) | Macro section of the research report |
| **W2** | Markets, instruments & stock basics | S5 Instruments, venues, clearing & settlement · S6 Stock market mechanics: orders, sessions, auctions, corporate actions · S7 Fundamentals & valuation primer · S8 Microstructure I: order books & market structure | Index/ETF and corporate-actions lab | Instrument fact sheets; adjusted-price workbook |
| **W3** | Microstructure II & derivatives | S9 Microstructure II: spreads, impact, volatility controls · S10 Futures: carry, basis, margin, rolls · S11 Options: payoffs, parity, bounds, chains · S12 F&O basics & strategies I | Microstructure data lab (spreads from quotes and trades) | Microstructure section of the research report |
| **W4** | Strategies, Excel & integration | S13 F&O strategies II & choosing structures · S14 Excel primer I: data, returns, risk · S15 Excel primer II: futures and options models · S16 Integration: a trading day end to end & assessment | Excel workbook build and review | Final research report, Excel workbook, quiz |

---

## 4. Session-by-Session Plan

> Each session: **15 min recap/theory → 45 min worked examples → 20 min guided lab → 10 min wrap-up and homework.**
> Excel templates live in `labs/part01/excel/`; ready-to-run notebooks in `notebooks/part01/`.

### Week 1 — Orientation & Macroeconomics

#### S1 · Orientation: The Algorithmic Trading Landscape (item 16)

| Block | Content |
|---|---|
| Theory | **Who trades and why:** market makers and HFT firms (liquidity provision), trend followers/CTAs, statistical arbitrage, multi-strategy "pod" funds, options market makers, long-only institutions (who move markets through rebalancing and flows), retail. **Strategy families** (a map of the whole program): momentum, mean reversion, carry, volatility, statistical arbitrage, event-driven, market making, ML-driven. **Infrastructure:** data → research → backtest → validation → execution → risk → monitoring (the `quantforge` platform built from Month 3). What an algorithmic edge is and why most edges are small, crowded and decaying. Realistic expectations about returns, costs and risk. |
| Worked examples | Tour of an IB and an Alpaca paper account: quotes, contract details, option chain, account/margin screens. |
| Lab | Open both paper accounts; locate the contract specifications for SPY, ES, EURUSD and an SPY option; record them in the research log. |
| Homework | One page: "Which market participant would be on the other side of my first strategy idea, and why would they trade with me?" |

#### S2 · Macro I: Growth, Inflation, Employment & Data Releases (1.1)

| Block | Content |
|---|---|
| Theory | **Growth:** GDP (advance, second, third estimates), ISM manufacturing and services PMIs, retail sales, industrial production, jobless claims. **Inflation:** CPI and core CPI, PCE and core PCE (the Fed's target measure), PPI, wage growth, inflation expectations. **Employment:** nonfarm payrolls, unemployment rate, participation, average hourly earnings. **How markets react:** to the **surprise** (actual − consensus), not the level; reaction depends on the regime (in high inflation, good growth news can be bad for stocks). **Release mechanics:** most US releases at 08:30 ET; the economic calendar; embargo and release-time precision. **Revisions and point-in-time data:** first prints are revised (payrolls, GDP); a backtest must use the value **known on that date**, available from ALFRED vintages. |
| Worked examples | Compare the first-released and the latest revised values of payrolls for 2020–2024 (ALFRED); show how a strategy using revised data would "know" information nobody had. |
| Lab | Build an Excel release calendar for the next 4 weeks with consensus fields; download 10 years of CPI, core PCE, unemployment and GDP (provided notebook → CSV → Excel). |
| Homework | For 3 recent CPI releases, record consensus, actual, surprise, and the S&P 500 and 2-year Treasury moves in the first 30 minutes. |

#### S3 · Macro II: Central Banks, Rates & the Yield Curve (1.1)

| Block | Content |
|---|---|
| Theory | **Federal Reserve:** dual mandate, FOMC (8 scheduled meetings a year; statement at 14:00 ET, press conference at 14:30 ET), the dot plot, forward guidance, the policy rate range and the effective fed funds rate (EFFR), QE/QT and the balance sheet, liquidity (bank reserves, reverse repo facility, Treasury General Account). **Rates:** Treasury bills, notes and bonds; yield vs price; the **yield curve** (2s10s, 3m10y), inversions and recessions (a signal, not a clock); real yields (TIPS) and breakeven inflation; term premium. **Market-implied policy:** fed funds futures settle on 100 − the **monthly average** EFFR, so the post-meeting rate can be backed out (below). Other central banks that move US markets: ECB, BoJ, PBoC. |
| Worked examples | Implied post-meeting rate and cut/hike probability from a fed funds futures price (formula below); yield-curve chart with recession shading. |
| Lab | Excel sheet computing implied probabilities for the next two FOMC meetings from futures prices; compare with published market-implied probabilities. |
| Homework | Summarize the last FOMC statement: what changed in the wording vs the previous statement, and how the 2-year yield and the dollar reacted. |

**Fed funds futures, worked example.** Futures price 95.73 → implied **average** EFFR for the month = 100 − 95.73 = 4.27%. Current EFFR 4.33%; the FOMC meets on day 17 of a 30-day month (new rate effective from day 18).

`post-meeting rate = (average × days_in_month − current_rate × days_before) / days_after = (4.27 × 30 − 4.33 × 17) / 13 ≈ 4.19%`

`probability of a 25 bp cut ≈ (4.33 − 4.19) / 0.25 ≈ 55%`

| Excel (sheet `FedFunds`) | Formula |
|---|---|
| B1 futures price, B2 current rate, B3 meeting day, B4 days in month | inputs |
| B5 post-meeting rate | `=((100-B1)*B4-B2*B3)/(B4-B3)` → 4.19 |
| B6 probability of a 25 bp cut | `=(B2-B5)/0.25` → 0.55 |

#### S4 · Macro III: Business Cycle, the Dollar, Commodities & Intermarket Analysis (1.1)

| Block | Content |
|---|---|
| Theory | **Business cycle** phases (expansion, slowdown, recession, recovery) and typical sector leadership; leading vs coincident vs lagging indicators. **The US dollar:** interest-rate differentials, risk-off demand, trade balance; the DXY index. **Commodities:** oil (supply, OPEC+, inventories), gold (real yields, dollar, central-bank buying), copper (growth). **Credit:** investment-grade and high-yield spreads as a stress gauge. **Intermarket relationships** (bonds ↔ equities ↔ dollar ↔ commodities) and why they are **not stable**: the stock–bond correlation was mostly negative from the late 1990s to 2021 and turned positive in 2022 when inflation dominated. **Regimes:** growth up/down × inflation up/down as a simple 4-quadrant framework. |
| Worked examples | Rolling 1-year correlation of S&P 500 and 10-year Treasury returns, 2000–2025; sector performance by regime quadrant. |
| Lab | **Macro regime dashboard** (clinic W1): growth and inflation trend indicators, yield curve, credit spread, dollar, with the current regime classified by explicit rules. |
| Homework | Write the macro section of the research report (current regime, key risks, what would change the view). |

**Clinic W1 (120 min):** macro regime dashboard in Excel (with the provided notebook to refresh data), using **point-in-time** values for the history where vintages exist.

---

### Week 2 — Markets, Instruments & Stock Basics

#### S5 · Financial Markets & Instruments, Venues, Clearing & Settlement (1.2, items 7, 9)

| Block | Content |
|---|---|
| Theory | **Equities** (common/preferred, share classes, ADRs). **ETFs:** creation/redemption by authorized participants, NAV vs market price, premium/discount, leveraged and inverse ETFs (daily reset and volatility decay). **Indices:** construction (float-adjusted market-cap weighted like the S&P 500, price-weighted like the Dow, equal-weighted), divisor adjustments, rebalancing and index-inclusion effects, price vs total-return indices. **FX:** spot, currency pairs, base/quote, pips, sessions (Asia, London, New York), carry. **Bonds:** price–yield relationship, duration as rate sensitivity. **Futures, options, crypto** (overview; details in W3). **Venues:** NYSE, Nasdaq, Cboe, CME Group, IB's IDEALPRO for FX, crypto exchanges. **Clearing and settlement:** central counterparties (NSCC for US equities, OCC for listed options, CME Clearing for futures); US equities settle **T+1** (since May 2024). **Margin and shorting:** Reg T initial margin 50%, FINRA minimum maintenance 25% (brokers often require more), portfolio margin; short selling needs a locate and a borrow; borrow fees and recalls; Reg SHO. Broker account rules such as pattern-day-trader status (check your broker's current policy). |
| Worked examples | ETF premium `= (price − NAV) / NAV`; a cap-weighted index with a divisor change after a constituent replacement; margin-call price for a leveraged long (below). |
| Lab | Instrument fact sheets for SPY, QQQ, a leveraged ETF, ES, EURUSD and an SPY option: exchange, trading hours, tick size, multiplier, settlement, margin. |
| Homework | Simulate 60 days of a 2× daily leveraged ETF vs 2× the index return in a choppy market; explain the gap. |

**Margin-call price (long on margin):** buy 100 shares at $100 with 50% initial margin → loan $5,000. With 25% maintenance, a margin call occurs when `equity / value < 25%`, i.e. at price `P = loan / (shares × (1 − 0.25)) = 5,000 / 75 ≈ $66.67`.

#### S6 · Stock Market Mechanics: Orders, Sessions, Auctions, Corporate Actions (1.3, item 11)

| Block | Content |
|---|---|
| Theory | **Order types:** market, limit, stop, stop-limit, trailing stop, market-on-open/close (MOO/MOC), limit-on-close; time in force (DAY, GTC, IOC, FOK, OPG/CLS). **Sessions:** pre-market (from 04:00 ET), regular (09:30–16:00 ET), after-hours (to 20:00 ET); extended-hours liquidity and wider spreads; broker-specific overnight sessions. **Auctions:** opening and closing crosses, imbalance information before the close, why so much volume trades at the close (index funds). **Corporate actions:** splits and reverse splits, cash and stock dividends (declaration, ex-date, record date, pay date; with T+1, the ex-date is normally the same day as the record date), spin-offs, mergers, symbol changes, delistings. **Adjusted prices:** split and dividend adjustment factors; why backtests on unadjusted prices show fake crashes and why fully adjusted prices can distort level-based rules. |
| Worked examples | Build split- and dividend-adjusted series by hand for a stock with a 4-for-1 split and quarterly dividends; compare with a vendor's adjusted series. |
| Lab | Corporate-actions workbook: for 3 stocks, list 5 years of actions from filings/press releases and reproduce the adjusted price series. |
| Homework | Place (on paper) one of each order type on SPY via IB and Alpaca; record how each broker displays and handles it. |

#### S7 · Fundamentals & Valuation Primer (1.3, item 11)

| Block | Content |
|---|---|
| Theory | **Financial statements:** income statement, balance sheet, cash-flow statement, and how they connect. **Key ratios:** gross/operating/net margins, ROE, ROIC, leverage (debt/EBITDA), liquidity (current ratio), free cash flow, share count and dilution. **Valuation:** P/E, EV/EBITDA, EV/sales, FCF yield; a simple DCF and its sensitivity to the discount rate and terminal growth. **Earnings season:** EPS and revenue surprise vs consensus, guidance, the reaction on the next open (links to Part 11 event studies). **Filings:** 10-K, 10-Q, 8-K on SEC EDGAR; filing dates for point-in-time use (fundamentals are only known after they are filed). **Sectors:** the 11 GICS sectors and why sector exposure matters for risk. |
| Worked examples | Read a 10-K: locate revenue, operating income, free cash flow, share count and the risk-factors section; compute 5 ratios. |
| Lab | Excel comparison of 5 companies in one sector (margins, growth, leverage, valuation multiples) with filing dates recorded. |
| Homework | 1-page DCF for one company with a sensitivity table (discount rate × terminal growth). |

#### S8 · Market Microstructure I: Order Books & Market Structure (1.4, item 8)

| Block | Content |
|---|---|
| Theory | **Limit order book:** bids, asks, depth, price-time priority, how a market order "walks the book". **Quotes and NBBO:** national best bid and offer; the Reg NMS order-protection rule (trade-throughs), access-fee caps. **A fragmented market:** exchanges, alternative trading systems and dark pools, retail wholesalers and internalization, **payment for order flow** (how commission-free brokers such as Alpaca are paid). **Fees:** maker–taker vs inverted pricing. **Tick sizes:** $0.01 for most stocks priced at $1 or more (the SEC's 2024 Reg NMS amendments added a half-penny tick for certain tightly quoted stocks; check the current status). **Lot sizes** and odd lots. |
| Worked examples | A market order walking a 5-level book: average fill price and cost vs the mid. |
| Lab | Using a provided L2 snapshot, compute cost to buy 1,000 / 10,000 / 50,000 shares; plot cost vs size. |
| Homework | Read the Rule 606 order-routing report of one broker; summarize where orders go and what payments are received. |

**Clinic W2 (120 min):** index & ETF lab (reconstruct a small cap-weighted index with a constituent change; ETF premium/discount history) and corporate-actions reconciliation.

---

### Week 3 — Microstructure II & Derivatives

#### S9 · Market Microstructure II: Spreads, Impact & Volatility Controls (1.4, item 8)

| Block | Content |
|---|---|
| Theory | **Why spreads exist:** order-processing costs, inventory risk, **adverse selection** (trading against better-informed traders). Models, intuitively: **Glosten–Milgrom** (the spread protects the market maker from informed flow), **Kyle** (informed traders hide in noise; price impact λ), **Roll** (bid-ask bounce creates negative autocorrelation in price changes). **Spread measures:** quoted spread, **effective spread** `2 × |price − mid|`, **realized spread** (vs the mid a few minutes later) and price impact. **Market impact** grows roughly with the square root of order size relative to daily volume (details in Part 8). **High-frequency trading:** liquidity provision, latency arbitrage, and the debate. **Volatility controls:** limit up–limit down (LULD) single-stock bands; market-wide circuit breakers at 7%, 13% and 20% declines in the S&P 500. **Case studies:** the May 6, 2010 flash crash; the August 24, 2015 open. |
| Worked examples | Roll estimator `spread ≈ 2·√(−cov(Δpₜ, Δpₜ₋₁))` on simulated bid-ask bounce (a true spread of 0.10 is recovered as ≈ 0.098); effective vs quoted spread on trade data. |
| Lab | **Microstructure data lab** (clinic W3): for 5 stocks of different liquidity, compute quoted, effective and Roll spreads and intraday spread patterns (wider at the open, narrower midday). |
| Homework | Write the microstructure section of the research report. |

#### S10 · Futures: Cost of Carry, Basis, Margin & Rolls (1.5, item 13)

| Block | Content |
|---|---|
| Theory | Forwards vs futures (standardization, exchange trading, central clearing, daily settlement). **Cost of carry:** `F = S·e^{(r − q)T}` for equity indices (q = dividend yield); storage and convenience yield for commodities; **basis** = F − S and convergence at expiry; **contango** and **backwardation**, roll yield. **Margin:** initial and maintenance margin, **daily variation margin** (mark-to-market cash flows), leverage. **Contract specifications:** ES ($50 × index), MES ($5), NQ ($20), CL (1,000 barrels), GC (100 oz), ZN ($100,000 face), 6E (€125,000); tick size and tick value; month codes (H, M, U, Z for quarterly). **Expiry and rolls:** last trading day, first notice day for physical delivery, cash vs physical settlement, when volume moves to the next contract. |
| Worked examples | ES fair value: S = 6,000, r = 4.3%, q = 1.3%, 80 days → `F = 6,000·e^{(0.043 − 0.013)·80/365} ≈ 6,039.58`; basis ≈ 39.6 points. Daily mark-to-market ledger for a long ES position over 10 days. |
| Lab | Excel: futures fair value vs market price for ES and NQ; a 10-day variation-margin ledger with a margin call. |
| Homework | Plot the CL futures curve on two dates (one in contango, one in backwardation); explain the difference. |

#### S11 · Options: Payoffs, Put-Call Parity, Bounds & Option Chains (1.5, item 13)

| Block | Content |
|---|---|
| Theory | Calls and puts; long vs short; **moneyness** (ITM, ATM, OTM); intrinsic and time value; American vs European exercise; exercise, assignment and settlement (SPX: European, cash-settled; SPY: American, physically settled); multiplier (100 for US equity options). **Put-call parity** (European, with dividend yield): `C − P = S·e^{−qT} − K·e^{−rT}`, and what a violation would mean. **No-arbitrage bounds** (e.g. a call is worth at least `S·e^{−qT} − K·e^{−rT}` and at most `S·e^{−qT}`). **What drives option value** and in which direction: spot, strike, time, volatility, rates, dividends (intuition for the Greeks, formalized in Part 6). **Implied volatility** as the market's price quote; the **VIX** (30-day implied volatility of S&P 500 options). **Reading an option chain:** bid/ask, last, volume, open interest, IV, Greeks; OCC symbols (e.g. `SPY261218C00600000`). |
| Worked examples | Parity check on a live SPY chain (American options: expect small deviations from dividends and early exercise); payoff tables for the four basic positions. |
| Lab | From an SPX chain snapshot: find ATM strike, compute intrinsic/time value for 10 strikes, and check parity at 5 strikes. |
| Homework | Explain in half a page why deep ITM American puts can be worth exercising early and American calls on non-dividend stocks are not. |

#### S12 · F&O Basics & Strategies I (item 10, 1.6, item 14)

| Block | Content |
|---|---|
| Theory | Building blocks: every strategy is a sum of payoffs (long/short call, put, underlying, futures). **Stock + option:** covered call, protective put, collar. **Vertical spreads:** bull call, bear put (debit), bull put, bear call (credit); max profit, max loss, breakeven; trading off cost against capped upside. **Synthetics:** synthetic long (long call + short put), conversions and reversals (links to parity). **Futures strategies:** hedging a stock portfolio with index futures (beta-adjusted), calendar spreads. **Risk profile language:** defined vs undefined risk; margin for spreads vs naked options. |
| Worked examples | Payoff and P&L tables for a covered call, a collar and a bull call spread; the synthetic-long equivalence shown numerically. |
| Lab | Excel payoff builder: any combination of up to 4 legs, with max profit, max loss and breakevens (formula approach from S15). |
| Homework | For each of 6 strategies, write: view (direction, volatility, time), max profit, max loss, breakeven(s), and the main risk. |

**Clinic W3 (120 min):** microstructure data lab (spreads from quotes and trades, intraday patterns) and completion of the microstructure section of the report.

---

### Week 4 — Strategies, Excel & Integration

#### S13 · F&O Strategies II & Choosing a Structure (1.6, item 14)

| Block | Content |
|---|---|
| Theory | **Volatility views:** long/short straddle and strangle; butterflies and iron butterflies; iron condors. **Time views:** calendar and diagonal spreads. **Ratio spreads and backspreads** (and their tail risk). **Choosing a structure:** a matrix of *direction* (up/down/neutral) × *volatility* (expected to rise/fall) × *time* (short/long horizon), plus the current level of implied volatility (cheap vs expensive). **Futures and options together:** hedging a portfolio with index puts vs short futures; collars on concentrated positions. Why short-premium strategies have high win rates and occasional large losses (preview of Parts 7–8). |
| Worked examples | Payoff diagrams at expiry for straddle, strangle, butterfly, iron condor and calendar (calendar: P&L at the front expiry depends on the back option's value, a first use of pricing). |
| Lab | Given 6 market scenarios (view + IV level), choose and justify a structure; build it in the payoff builder. |
| Homework | Stress each chosen structure: P&L if the underlying gaps ±10% overnight. |

#### S14 · Excel Primer I: Data, Returns & Risk (1.7, item 12)

| Block | Content |
|---|---|
| Theory | **Model hygiene:** separate inputs, calculations and outputs; named ranges; no hard-coded numbers inside formulas; consistent units; a version and source note on every sheet. **Data:** import CSV from brokers and FRED; Excel Tables; `XLOOKUP`; dates and time zones (ET vs UTC). **Returns:** simple `=B3/B2-1` and log `=LN(B3/B2)`; cumulative return; CAGR; annualized volatility; drawdown and max drawdown; rolling metrics; correlation matrix; pivot tables by year and month; charts. |
| Worked examples | Returns sheet (formulas below) on 10 years of SPY, TLT and GLD data. |
| Lab | Build the risk dashboard: CAGR, volatility, max drawdown, best/worst month, correlation, year-by-month return table. |
| Homework | Add a 60-day rolling volatility and rolling correlation chart. |

| Measure (prices in column A from row 2, log returns in column B) | Excel formula |
|---|---|
| Log return (row 3) | `=LN(A3/A2)` |
| Drawdown (row n) | `=A2/MAX(A$2:A2)-1`, filled down |
| Max drawdown | `=MIN(C2:C2521)` |
| Annualized volatility (daily data) | `=STDEV.S(B3:B2521)*SQRT(252)` |
| CAGR (daily data) | `=(A2521/A2)^(252/(COUNT(A2:A2521)-1))-1` |

#### S15 · Excel Primer II: Futures & Options Models (1.7, item 12)

| Block | Content |
|---|---|
| Theory | **Black–Scholes–Merton in Excel** (European options with a continuous dividend yield) and its assumptions (the full treatment is Part 6). Greeks from closed forms: delta, gamma, vega (per volatility point), theta (per calendar day). **Scenario analysis:** data tables for spot × volatility P&L grids. **Goal Seek / Solver:** implied volatility from a market price; a two-asset minimum-variance weight. **Futures models:** fair value, basis, variation-margin ledger. **Bridge to Python:** how the same models will be rebuilt in code in Part 3 (reading and writing Excel with `pandas.read_excel` / `openpyxl`), and why code replaces spreadsheets for anything automated. |
| Worked examples | The BSM sheet below; implied vol with Goal Seek; P&L grid for a bull call spread. |
| Lab | Complete the graded workbook (Section 7). |
| Homework | Check the sheet: put-call parity error must be 0; call price must rise with volatility and time; compare one price with the broker's model price and explain differences. |

**Black–Scholes–Merton sheet (named inputs: `Spot`, `Strike`, `Years`, `Rate`, `Div`, `Vol`; outputs also named).** Excel treats names like `R`, `C` or `D1` as references, so the names below are chosen to avoid that.

| Name | Formula |
|---|---|
| `D_one` | `=(LN(Spot/Strike)+(Rate-Div+Vol^2/2)*Years)/(Vol*SQRT(Years))` |
| `D_two` | `=D_one-Vol*SQRT(Years)` |
| `Call` | `=Spot*EXP(-Div*Years)*NORM.S.DIST(D_one,TRUE)-Strike*EXP(-Rate*Years)*NORM.S.DIST(D_two,TRUE)` |
| `Put` | `=Strike*EXP(-Rate*Years)*NORM.S.DIST(-D_two,TRUE)-Spot*EXP(-Div*Years)*NORM.S.DIST(-D_one,TRUE)` |
| `DeltaCall` | `=EXP(-Div*Years)*NORM.S.DIST(D_one,TRUE)` |
| `Gamma` | `=EXP(-Div*Years)*NORM.S.DIST(D_one,FALSE)/(Spot*Vol*SQRT(Years))` |
| `VegaPt` (per 1 vol point) | `=Spot*EXP(-Div*Years)*NORM.S.DIST(D_one,FALSE)*SQRT(Years)/100` |
| `ThetaCallDay` (per calendar day) | `=(-Spot*EXP(-Div*Years)*NORM.S.DIST(D_one,FALSE)*Vol/(2*SQRT(Years))-Rate*Strike*EXP(-Rate*Years)*NORM.S.DIST(D_two,TRUE)+Div*Spot*EXP(-Div*Years)*NORM.S.DIST(D_one,TRUE))/365` |
| `ParityErr` (must be 0) | `=Call-Put-(Spot*EXP(-Div*Years)-Strike*EXP(-Rate*Years))` |
| Long-call P&L at expiry for price in `D2` | `=MAX(D2-Strike,0)-Call` |

> Verified: with Spot 100, Strike 105, 0.5 years, Rate 4%, Div 1%, Vol 25%, the sheet (evaluated with the `formulas` Excel engine) gives Call 5.5482, Put 8.9678, Delta 0.4568, Gamma 0.02234, Vega 0.2792 per vol point, Theta −0.0223 per day and a parity error of exactly 0, identical to an independent Python implementation. The returns and fed-funds formulas above were checked the same way.

#### S16 · Integration: A Trading Day End to End & Assessment

| Block | Content |
|---|---|
| Theory | **Walk through one US trading day** as an algorithmic trader sees it: overnight futures and FX, 08:30 ET data release and the surprise, pre-market, the opening auction, first-hour liquidity and spreads, midday, FOMC at 14:00 ET on meeting days, closing-imbalance information and the closing auction, after-hours earnings, settlement on T+1. At each step: what data exists, when it becomes known, what it costs to trade, and which Part of the program deals with it. |
| Activity | Research-report presentations (10 min each); quiz; Excel workbook review. |
| Homework | Prepare for Part 2 (statistics) and Part 3 (Python): install the course environment. |

**Clinic W4 (120 min):** Excel workbook build and peer review against the model-hygiene checklist.

---

## 5. Lab & Notebook Map

| Artifact | Session | Location | Used again in |
|---|---|---|---|
| Release calendar & macro data | S2 | `labs/part01/excel/macro_calendar.xlsx`, `notebooks/part01/01_fred_alfred.ipynb` | Part 5 (sentiment, point-in-time), Part 11 (events) |
| Fed funds implied probabilities | S3 | `labs/part01/excel/fed_funds.xlsx` | Part 11 (FOMC events) |
| Macro regime dashboard | S4 | `labs/part01/excel/macro_regime.xlsx`, `notebooks/part01/02_macro_dashboard.ipynb` | Parts 7, 10 (regimes) |
| Instrument fact sheets | S5 | `labs/part01/instruments.md` | Part 4 (contract mapping) |
| Corporate actions & adjusted prices | S6 | `labs/part01/excel/corporate_actions.xlsx` | Part 4 (data cleaning), Part 8 (biases) |
| Fundamentals comparison & DCF | S7 | `labs/part01/excel/fundamentals.xlsx` | Part 9 (factors), Part 11 (extraction) |
| Order-book cost & spread measures | S8–S9 | `labs/part01/excel/microstructure.xlsx`, `notebooks/part01/03_spreads.ipynb` | Part 8 (costs), Part 12 (execution) |
| Futures fair value & margin ledger | S10 | `labs/part01/excel/futures.xlsx` | Part 4 (futures), Part 6 (futures library) |
| Payoff builder | S12–S13 | `labs/part01/excel/payoff_builder.xlsx` | Part 7 (option strategy builder) |
| Returns & risk dashboard | S14 | `labs/part01/excel/returns_risk.xlsx` | Part 2 (statistics), Part 8 (performance) |
| BSM & Greeks sheet | S15 | `labs/part01/excel/bsm_greeks.xlsx` | Part 6 (pricing engine validation) |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Using revised macro data in historical analysis | Indicators look more predictive than they were | ALFRED vintages; record the release date with every value |
| 2 | Reacting to the level of a release, not the surprise | Confusing results in event analysis | Always compute actual − consensus |
| 3 | Treating correlations (e.g. stocks vs bonds) as fixed | Hedges fail when the regime changes | Rolling correlations; regime labels |
| 4 | Ignoring corporate actions | Fake crashes and jumps in price history | Adjusted prices with documented adjustment factors |
| 5 | Using fundamentals before their filing date | Look-ahead in valuation screens | Store and use filing dates |
| 6 | Assuming you trade at the last price or the mid | Costs underestimated | Quoted/effective spread; walk the book for size |
| 7 | Wrong contract multiplier or tick value | P&L off by 50× or 100× | Fact sheet from exchange/broker specs |
| 8 | Ignoring futures margin cash flows | Surprise margin calls | Daily variation-margin ledger |
| 9 | Applying European formulas to American options without thought | Parity "violations" that are not arbitrage | Account for dividends and early exercise |
| 10 | Mixing units (vol in % vs decimal, theta per year vs per day) | Greeks 100× or 365× off | Units column in every sheet; named inputs |
| 11 | Hard-coded numbers inside Excel formulas | Unauditable, error-prone models | Inputs/calculations/outputs separation; named ranges |
| 12 | Using Excel names that clash with cell references (`R`, `C`, `D1`) | `#NAME?` or wrong references | Descriptive names such as `Rate`, `D_one` |
| 13 | Believing high-win-rate option selling is low risk | Occasional large losses | Always show max loss and a gap scenario |
| 14 | Mixing ET and UTC timestamps | Events aligned to the wrong bars | One time zone per dataset, labelled |

---

## 7. Assessment

**A. Research report (≤ 10 pages), 45%.**
1. *Macro:* current regime by explicit rules; policy expectations from fed funds futures; yield curve and credit conditions; three scenarios with market implications.
2. *Microstructure:* spread and cost-of-trading study for 5 stocks of different liquidity (quoted, effective, Roll spreads; intraday pattern; cost to trade 1% of daily volume).
3. *Instruments:* how you would express one macro view through a stock/ETF, a futures contract and an options structure, with costs and risks compared.

**B. Excel workbook, 40%.** Returns and risk dashboard; futures fair value and variation-margin ledger; payoff builder (up to 4 legs); BSM price and Greeks sheet with parity check and a spot × volatility P&L grid; fed funds probability sheet. Graded on correctness **and** model hygiene.

**C. Quiz, 15%.** 30 questions on releases and central banks, order types and auctions, corporate actions, market structure, futures and options mechanics.

| Criterion | Points |
|---|---|
| Macro analysis: correct interpretation, point-in-time awareness, market-implied expectations | 20 |
| Microstructure study: correct spread measures, sound conclusions on trading costs | 15 |
| Instrument comparison: correct mechanics, costs and risks | 10 |
| Excel models: correctness (BSM, Greeks, parity, futures, payoffs, returns) | 25 |
| Excel model hygiene (inputs/calcs/outputs, names, units, documentation) | 15 |
| Quiz | 15 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the BSM sheet's parity error must be 0 and the historical macro analysis must use point-in-time data where vintages exist.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | Mishkin, F. (2021). *The Economics of Money, Banking, and Financial Markets* (13th ed.). Pearson. |
| Book | Harris, L. (2003). *Trading and Exchanges: Market Microstructure for Practitioners*. Oxford University Press. |
| Book | Bouchaud, J.-P., Bonart, J., Donier, J. & Gould, M. (2018). *Trades, Quotes and Prices*. Cambridge University Press. |
| Book | O'Hara, M. (1995). *Market Microstructure Theory*. Blackwell. |
| Book | Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.). Pearson. |
| Book | Natenberg, S. (2014). *Option Volatility and Pricing* (2nd ed.). McGraw-Hill. |
| Book | Damodaran, A. (2012). *Investment Valuation* (3rd ed.). Wiley. |
| Book | Ilmanen, A. (2011). *Expected Returns*. Wiley. |
| Paper | Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." *Journal of Finance*, 39(4). |
| Paper | Glosten, L. & Milgrom, P. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *JFE*, 14(1). |
| Paper | Kyle, A. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6). |
| Paper | Estrella, A. & Mishkin, F. (1998). "Predicting U.S. Recessions: Financial Variables as Leading Indicators." *Review of Economics and Statistics*, 80(1). |
| Paper | Kirilenko, A., Kyle, A., Samadi, M. & Tuzun, T. (2017). "The Flash Crash: High-Frequency Trading in an Electronic Market." *Journal of Finance*, 72(3). |
| Report | CFTC & SEC (2010). *Findings Regarding the Market Events of May 6, 2010*. |
| Docs | Federal Reserve (FOMC statements, calendars), BLS and BEA release schedules, FRED/ALFRED, SEC EDGAR, CME Group contract specifications, OCC options symbology, IB and Alpaca order-type documentation |

---

## 9. Instructor Notes

- Keep US market times in **ET** throughout, and state the UTC equivalent for anything that will later be stored in the platform (Part 4 stores everything in UTC).
- Run S2 or S3 live on a real release or FOMC day if the calendar allows; otherwise replay a recorded one minute by minute.
- Provide the notebooks fully written; learners should only change inputs this month. The goal is to see what data looks like before learning to code it in Part 3.
- Macro and market-structure rules change (tick sizes, settlement cycles, broker account rules, margin requirements). Keep a "last verified" date on every rule stated in the slides and re-check before each cohort.
- Nothing is traded with real money in Part 1; paper accounts are used only to explore quotes, contract details and order tickets.
