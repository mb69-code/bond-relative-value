
# Bond Relative Value — Euro SSA & Repo Desk

> End-to-end fixed income relative value framework for a Euro SSA repo desk:
> market-implied repo funding analysis, Z-spread and ASW modelling,
> and a synthetic Bund CTD specialness model — with a live Streamlit dashboard.

## Live Dashboard

**[Launch the interactive dashboard](https://repo-bonds-trading.streamlit.app)**

No installation required, it runs directly in your browser !

---

## Overview

This project models **two complementary views of the euro repo market**,
intentionally kept separate to reflect how a rates/repo desk actually
thinks about fixed income relative value:

| Framework | Notebook | Purpose |
|---|---|---|
| **Market-implied repo** | `repo_funding_model` | SSA carry, break-even funding, Z-spread & ASW vs Bund curve |
| **Specialness-driven repo** | `specialness_model` | CTD identification, OU specialness model, basis trade P&L |

The two frameworks answer different questions:
- *"Is this SSA bond cheap enough to carry vs its funding cost?"* → Framework 1
- *"Why is the Bund CTD trading special in repo, and how much carry does it generate?"* → Framework 2

---

## Repository Structure

```
bond-relative-value/
├── notebooks/
│   ├── repo_funding_model.ipynb     # Framework 1 — SSA funding & RV
│   └── specialness_model.ipynb      # Framework 2 — CTD specialness
├── dashboard/
│   └── repo_dashboard.py            # Streamlit interactive monitor
├── data/                            # Generated charts & outputs
├── requirements.txt
└── README.md
```

---

## Methodology

### Framework 1 — Market-Implied Repo (`repo_funding_model.ipynb`)

**Objective:** Quantify the net carry and relative value of euro SSA bonds
(KfW, EIB) versus Bund benchmarks (Bobl 5Y, Bund 10Y), explicitly
accounting for repo funding costs.

**Bond universe** (prices sourced from Deutsche Börse XFRA & issuer
websites as of 21 Apr 2026):

| Bond | ISIN | Coupon | Maturity | Repo spread vs GC |
|---|---|---|---|---|
| KfW 5Y | XS3344416287 | 2.875% | Jun-31 | −2 bps |
| KfW 10Y | XS3326554261 | 4.680% | Mar-36 | −3 bps |
| EIB 5Y | EU000A4EPCA0 | 2.625% | Jun-31 | −3 bps |
| EIB 10Y | EU000A4EM8H8 | 3.000% | Jan-36 | −4 bps |
| Bobl 5Y | DE000BU25067 | 2.500% | Apr-31 | −8 bps |
| Bund 10Y | DE000BU2Z064 | 2.900% | Feb-36 | −12 bps |

**Modelling choices:**

**① GC repo proxy**
$$r_{GC}(t) = \text{ESTER}(t) - 10\text{bps}$$
Calibrated on ICMA European Repo Market Survey (Dec 2025, published
Mar 2026). ESTER sourced live from ECB API
(`EST/B.EU000A2X2A25.WT`).

**② Net carry (ACT/365 coupon · ACT/360 repo)**
$$\text{Net Carry} = \frac{C \cdot \Delta t}{365} - r_{repo} \cdot
\frac{P \cdot \Delta t}{360}$$

**③ Break-even repo rate** — maximum funding cost before carry
turns negative:
$$r_{BE} = \frac{C}{P} \times \frac{360}{365}$$

**④ Z-spread** — constant spread over the ECB AAA Bund curve
(Svensson model, 9 tenors, cubic spline interpolation) computed
via Brent's method:
$$P_{market} = \sum_i CF_i \cdot e^{-(r_{spot}(t_i) + z) \cdot t_i}$$

**⑤ Asset Swap Spread (ASW)** — spread over the EUR swap curve,
proxied as ECB AAA curve − 25 bps (consistent with the structural
negative swap spread documented post-2015 ECB QE):
$$ASW = \frac{(C - r_{swap}) \cdot A + (1 - P/100)}{A}$$

>  ASW requires live Bloomberg EUSA quotes for production use.
> The proxy introduces a ~20 bps systematic bias — Z-spread is
> the reliable signal in this implementation.

---

### Framework 2 — Specialness-Driven Repo (`specialness_model.ipynb`)

**Objective:** Model the repo specialness dynamics of the Bund CTD
(Cheapest-To-Deliver) for the FGBLM6 futures contract (delivery
June 2026), and compute the carry P&L of a specialness trade.

**Part I — CTD Identification**

Using official Eurex conversion factors (sourced from
`eurexchange.com/ex-en/data/clearing-files`) and real cash prices
from Deutsche Börse (21 Apr 2026), the CTD is identified as the bond
with the lowest net basis:

$$\text{Net Basis}_i = (P_i - F \times CF_i) -
(\text{coupon cash} - P_i^{dirty} \times r_{GC} \times t)$$

CTD identified: **DBR 2.5% Feb-35** (`DE000BU2Z049`),
stable under ±1pt futures shock and ±50 bps GC shock
(sensitivity grid: 99 scenarios, 100% CTD stability).

**Part II — Ornstein-Uhlenbeck Specialness Model**

Since live special repo rates are not publicly available, specialness
is modelled synthetically via a two-component framework:

*Deterministic mean term structure* — calibrated on Bund repo
empirical ranges (ICMA ERCC 2022, IMF WP/18/258, ECB WP 2065):
$$\mu(\tau) = \gamma + \alpha \cdot e^{-\beta\tau}$$

with $\alpha = 120$ bps (peak component), $\beta = 0.04$ day$^{-1}$
(acceleration speed), $\gamma = 2$ bps (far-from-expiry floor).

*Stochastic dynamics (Euler-Maruyama discretisation)*:
$$dS_t = \kappa(\mu(\tau_t) - S_t)dt + \sigma dW_t, \quad S_t \geq 0$$

with $\kappa = 0.15$ day$^{-1}$ (mean reversion),
$\sigma = 3$ bps$/\sqrt{\text{day}}$ (daily volatility).

*Carry P&L formula* — daily carry on notional $N$ at cash price $P$:
$$\text{Daily PnL} = S_t \times \frac{N \times P/100}{10{,}000} \times \frac{1}{360}$$

**Part III — Sensitivity & Stress Tests**

One-at-a-time OU parameter sensitivity (±20% on α, β, κ, σ) and
a CTD switch stress test across 4 macro/idiosyncratic scenarios.

---

## Key Results

### Framework 1 — Carry Analysis (30d horizon, €10M)

| Bond | Net carry (bps) | Break-even repo | Safety margin |
|---|---|---|---|
| KfW 10Y | +23.5 | 4.62% | **+281 bps** |
| EIB 10Y | +10.0 | 3.00% | **+122 bps** |
| Bund 10Y | +9.7 | 2.89% | **+117 bps** |
| KfW 5Y | +8.5 | 2.87% | **+103 bps** |

> KfW 10Y dominates on carry due to its 4.68% coupon issued at the
> 2024 rate peak — a structural outlier. For risk-adjusted RV,
> **EIB 10Y** (+10 bps carry, +122 bps safety margin, Z-spread +26 bps)
> offers the most balanced profile across all three metrics.

### Framework 2 — Specialness Carry Trade

| Metric | Value |
|---|---|
| CTD (FGBLM6) | DBR 2.5% Feb-35 (`DE000BU2Z049`) |
| Current specialness (50d to expiry) | ~18 bps |
| Projected at delivery | ~100 bps (mean), 93–108 bps (P10/P90) |
| Trade: Entry 50d → Exit 10d, €10M | Mean P&L **€3,862** |
| P10 / P90 | €3,455 / €4,267 |
| Most sensitive parameter | α (±20% → ±€724 P&L impact) |

---

## Interactive Dashboard

Four tabs covering both frameworks:

- **SSA Monitor** — Live Z-spreads vs ECB AAA curve, bond universe table
- **Repo Funding** — Net carry decomposition, break-even repo, stress test
- **CTD Specialness** — OU model visualisation, three-level repo bridge
  (GC / SSA / CTD implied repo), carry P&L fan chart
- **RV Matrix** — Composite ranking across Z-spread, carry, safety margin

```bash
pip install -r requirements.txt
streamlit run dashboard/repo_dashboard.py
```

---

## Getting Started

```bash
git clone https://github.com/mb69-code/bond-relative-value
cd bond-relative-value
pip install -r requirements.txt
jupyter notebook notebooks/repo_funding_model.ipynb
```

---

## References

- Duffie, D. (1996). *Special repo rates*. Journal of Finance, 51(2).
- Buraschi, A. & Menini, D. (2002). *Liquidity risk and specialness*.
  Journal of Financial Economics, 64(2), 243–284.
- Arrata, W. et al. (2020). *The scarcity effect of QE on repo rates*.
  IMF Working Paper WP/18/258.
- Corradin, S. & Maddaloni, A. (2020). *The importance of being special*.
  ECB Working Paper No. 2065.
- Hill, A. (2022). *r is not a constant*. ICMA ERCC.
- ICMA (2026). *European Repo Market Survey No. 50*. March 2026.
- ECB Statistical Data Warehouse — data-api.ecb.europa.eu
- Eurex — Bund Future contract specifications & conversion factors
  
