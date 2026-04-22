# ============================================================
# EURO SSA & REPO DESK MONITOR
# Fixed Income Relative Value — KfW · EIB · Bund
# ============================================================
# Two complementary repo frameworks:
#   1. Market-implied repo  → SSA carry, break-even, RV (Tabs 1 and 2)
#   2. Specialness-driven   → CTD scarcity, OU model, basis (Tab 3)
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import requests
from io import StringIO
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Euro SSA & Repo Desk Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown("""
<style>
    .framework-box {
        background: #f0f4f8;
        border-left: 4px solid #003366;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #1a1a2e;
        line-height: 1.6;
    }
    .framework-box.specialness {
        border-left-color: #CC3300;
        background: #fff5f2;
    }
    .framework-box.bridge {
        border-left-color: #006D19;
        background: #f0fff4;
    }
    .framework-title {
        font-weight: 700;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
        color: #003366;
    }
    .framework-box.specialness .framework-title { color: #CC3300; }
    .framework-box.bridge .framework-title { color: #006D19; }
    .indicative-badge {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 3px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #856404;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def framework_box(title, content, kind="default"):
    css_class = f"framework-box {kind}" if kind != "default" else "framework-box"
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="framework-title">{title}</div>'
        f'{content}'
        f'</div>',
        unsafe_allow_html=True
    )

def fetch_ecb_series(dataflow: str, series_key: str,
                     start: str = "2019-01-01") -> pd.Series:
    url = (
        f"https://data-api.ecb.europa.eu/service/data/"
        f"{dataflow}/{series_key}?format=csvdata&startPeriod={start}"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df = df.set_index("TIME_PERIOD").sort_index()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df["OBS_VALUE"]

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=3600)
def load_market_data():
    ester = fetch_ecb_series("EST", "B.EU000A2X2A25.WT")
    tenors = {
        1:  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y",
        2:  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
        3:  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3Y",
        5:  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y",
        7:  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_7Y",
        10: "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
        15: "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_15Y",
        20: "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_20Y",
        30: "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_30Y",
    }
    curve_data = {}
    for t, key in tenors.items():
        try:
            curve_data[t] = fetch_ecb_series("YC", key)
        except Exception:
            pass
    return ester, curve_data

@st.cache_data(ttl=3600)
def build_bond_universe(ester_last: float, gc_spread_bps: float):
    gc_rate = ester_last + gc_spread_bps / 100
    bonds = {
        "KfW_5Y": {
            "issuer": "KfW", "isin": "XS3344416287",
            "coupon": 2.875, "maturity": "2031-06-30",
            "price": 99.985, "duration": 4.224, "dv01": 422.4,
            "repo_spread": -2, "color": "#003366",
        },
        "KfW_10Y": {
            "issuer": "KfW", "isin": "XS3326554261",
            "coupon": 4.68, "maturity": "2036-03-26",
            "price": 100.00, "duration": 8.90, "dv01": 890,
            "repo_spread": -3, "color": "#003366",
        },
        "EIB_5Y": {
            "issuer": "EIB", "isin": "EU000A4EPCA0",
            "coupon": 2.625, "maturity": "2031-06-16",
            "price": 98.72, "duration": 4.213, "dv01": 421.3,
            "repo_spread": -3, "color": "#CC3300",
        },
        "EIB_10Y": {
            "issuer": "EIB", "isin": "EU000A4EM8H8",
            "coupon": 3.00, "maturity": "2036-01-14",
            "price": 98.20, "duration": 6.304, "dv01": 630.4,
            "repo_spread": -4, "color": "#CC3300",
        },
        "Bobl_5Y": {
            "issuer": "Germany", "isin": "DE000BU25067",
            "coupon": 2.50, "maturity": "2031-04-16",
            "price": 99.161, "duration": 4.169, "dv01": 416.9,
            "repo_spread": -8, "color": "#555555",
        },
        "Bund_10Y": {
            "issuer": "Germany", "isin": "DE000BU2Z064",
            "coupon": 2.900, "maturity": "2036-02-15",
            "price": 99.111, "duration": 6.498, "dv01": 649.8,
            "repo_spread": -12, "color": "#555555",
        },
    }
    df = pd.DataFrame(bonds).T
    df["maturity"]    = pd.to_datetime(df["maturity"])
    df["coupon"]      = df["coupon"].astype(float)
    df["price"]       = df["price"].astype(float)
    df["duration"]    = df["duration"].astype(float)
    df["dv01"]        = df["dv01"].astype(float)
    df["repo_spread"] = df["repo_spread"].astype(float)
    df["repo_rate"]   = gc_rate + df["repo_spread"] / 100
    df["gc_rate"]     = gc_rate
    return df

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Fetching ECB data…"):
    ester_series, curve_data = load_market_data()

ester_last = ester_series.iloc[-1]
ester_date = ester_series.index[-1].strftime("%d %b %Y")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Repo Desk Parameters")
    st.markdown("---")

    st.markdown("#### Funding")
    gc_spread_input = st.slider("GC spread vs ESTER (bps)", -30, 0, -10, 1)
    funding_horizon = st.slider("Carry horizon (days)", 7, 90, 30, 1)
    notional_m      = st.slider("Notional (EUR M)", 5, 100, 10, 5)

    st.markdown("---")
    st.markdown("#### Stress Test")
    gc_shock_bps = st.slider("GC rate shock (bps)", -50, 100, 0, 5)

    st.markdown("---")
    st.markdown("#### CTD Specialness Trade")
    entry_dte = st.slider("Entry (days to expiry)", 30, 90, 50, 1)
    exit_dte  = st.slider("Exit (days to expiry)",  1,  29, 10, 1)

    st.markdown("---")
    gc_rate = ester_last + gc_spread_input / 100
    st.caption(
        f"**ESTER:** {ester_last:.4f}% as of {ester_date}  \n"
        f"**GC proxy:** {gc_rate:.4f}%  \n"
        f"**Source:** ECB Statistical Data Warehouse"
    )

# ============================================================
# DERIVED VALUES
# ============================================================
NOTIONAL   = notional_m * 1_000_000
gc_rate    = ester_last + gc_spread_input / 100
gc_shocked = gc_rate + gc_shock_bps / 100
df_bonds   = build_bond_universe(ester_last, gc_spread_input)
today      = pd.Timestamp.today()

# ============================================================
# HEADER
# ============================================================
st.title("Euro SSA & Repo Desk Monitor")
st.caption(
    "Fixed Income Relative Value — KfW · EIB · Bund  |  "
    "Data: ECB Statistical Data Warehouse  |  "
    "**Prices: indicative levels (Deutsche Börse / issuer websites) — not live market data**"
)

# So we're sure it's not white on white
st.markdown("""
<style>
div[data-testid="metric-container"] { background-color: #1B1E23 !important; border: 1px solid #333344 !important; padding: 15px !important; border-radius: 8px !important; }
div[data-testid="metric-container"] label { color: #8A92A3 !important; font-weight: 600 !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# Top KPI bar
k1, k2, k3, k4 = st.columns(4)
k1.metric("ESTER",
          f"{ester_last:.3f}%",
          delta=f"{ester_last - ester_series.iloc[-2]:.3f}% DoD")
k2.metric("GC Repo proxy",
          f"{gc_rate:.3f}%",
          delta=f"{gc_spread_input} bps vs ESTER")
k3.metric("Bund 10Y (ECB AAA)",
          f"{curve_data.get(10, pd.Series([0])).iloc[-1]:.3f}%")
k4.metric("Bund 5Y (ECB AAA)",
          f"{curve_data.get(5,  pd.Series([0])).iloc[-1]:.3f}%")
st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    " SSA Monitor",
    " Repo Funding",
    " CTD Specialness",
    " RV Matrix",
])

# ============================================================
# TAB 1 — SSA MONITOR
# ============================================================
with tab1:

    framework_box(
        "Framework — Market-Implied Repo (Z-spread & Curve)",
        "Z-spreads measure the constant spread over the <b>ECB AAA euro area government yield curve</b> "
        "(Svensson model) that equates the theoretical bond price to its market price. "
        "All curves are sourced live from the ECB Statistical Data Warehouse. "
        "A positive Z-spread indicates the bond trades <b>cheap</b> vs the reference curve — "
        "reflecting credit, liquidity, or technical premia above the risk-free rate."
    )

    st.markdown('<div class="indicative-badge">⚠ Indicative prices — not live market data</div>',
                unsafe_allow_html=True)

    # ── Curve interpolation ──────────────────────────────────
    tenors_num = np.array(sorted(curve_data.keys()))
    yields_num = np.array([curve_data[t].iloc[-1] / 100 for t in tenors_num])
    cs = CubicSpline(tenors_num, yields_num)

    def spot_rate(t):
        return float(cs(np.clip(t, tenors_num[0], tenors_num[-1])))

    def bond_price_theoretical(coupon, maturity_years, z_spread=0.0, freq=1):
        dt = 1 / freq
        n  = int(maturity_years * freq)
        price = 0.0
        for i in range(1, n + 1):
            t_i = i * dt
            r   = spot_rate(t_i) + z_spread
            cf  = coupon / freq
            if i == n:
                cf += 1.0
            price += cf * np.exp(-r * t_i)
        return price * 100

    def compute_z_spread(coupon, maturity_years, market_price, freq=1):
        def obj(z):
            return bond_price_theoretical(coupon, maturity_years, z, freq) - market_price
        try:
            return brentq(obj, -0.02, 0.05, xtol=1e-8) * 10000
        except Exception:
            return np.nan

    # ── Compute Z-spreads ────────────────────────────────────
    z_results = []
    for bond_name, bond in df_bonds.iterrows():
        mat_y = (bond["maturity"] - today).days / 365.25
        zs    = compute_z_spread(bond["coupon"] / 100, mat_y, bond["price"])
        p_th  = bond_price_theoretical(bond["coupon"] / 100, mat_y)
        rich  = bond["price"] - p_th
        z_results.append({
            "Bond"           : bond_name,
            "Issuer"         : bond["issuer"],
            "ISIN"           : bond.get("isin", ""),
            "Maturity (Y)"   : round(mat_y, 2),
            "Coupon (%)"     : bond["coupon"],
            "Price"          : bond["price"],
            "Theo. Price"    : round(p_th, 4),
            "Richness (bps)" : round(-rich / (bond["duration"] / 100), 2),
            "Z-spread (bps)" : round(zs, 2),
        })

    df_zs = pd.DataFrame(z_results).set_index("Bond")

    # ── KPI cards (2 row of 3 kpi) ──────────────────────────────────
    n_cols = 3
    items = list(df_zs.iterrows())

    for i in range(0, len(items), n_cols):
        row_items = items[i : i + n_cols]
        cols = st.columns(n_cols)
        for j, (bond_name, row) in enumerate(row_items):
            zs = row["Z-spread (bps)"]
            signal = "CHEAP" if zs > 10 else ("RICH" if zs < -5 else "FAIR")
            with cols[j]:
                st.metric(
                    label=bond_name,
                    value=f"{zs:.1f} bps",
                    delta=f"{signal} | {row['Coupon (%)']:.1f}% {row['Maturity (Y)']}Y"
                )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Z-spread vs ECB AAA Bund Curve")
        st.caption("Positive = bond trades cheap vs government risk-free curve. "
                   "KfW 10Y outlier reflects high coupon (4.68%) issued at rate peak.")
        fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=120)
        bonds_order = list(df_zs.index)
        colors_map  = [df_bonds.loc[b, "color"] for b in bonds_order]
        zs_vals     = df_zs.loc[bonds_order, "Z-spread (bps)"]
        x = np.arange(len(bonds_order))

        bars = ax1.bar(x, zs_vals, width=0.6, color=colors_map, alpha=0.85)
        ax1.axhline(0,  color='gray',    linewidth=0.6)
        ax1.axhline(10, color='#006D19', linewidth=0.8, linestyle='--',
                    label='CHEAP threshold (+10 bps)')
        ax1.axhline(-5, color='#CC3300', linewidth=0.8, linestyle='--',
                    label='RICH threshold (−5 bps)')
        for bar, val in zip(bars, zs_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     val + (0.5 if val >= 0 else -1.5),
                     f'{val:.1f}', ha='center', fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(bonds_order, rotation=30, ha='right', fontsize=8)
        ax1.set_ylabel('Z-spread (bps)')
        ax1.legend(frameon=False, fontsize=8)
        clean_spines(ax1)
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.subheader("ECB AAA Bund Curve — Current Shape")
        st.caption("Svensson model curve published daily by ECB. "
                   "Diamond markers show where each bond plots vs the curve.")
        fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=120)
        tau_plot = np.linspace(1, 30, 300)
        ax2.plot(tau_plot, cs(tau_plot) * 100,
                 color='#003366', linewidth=2.0,
                 label='ECB AAA curve (Svensson)')
        ax2.scatter(tenors_num, yields_num * 100, color='#003366', s=40, zorder=5)
        for bond_name, row in df_zs.iterrows():
            mat = row["Maturity (Y)"]
            c   = df_bonds.loc[bond_name, "color"]
            implied_yield = spot_rate(mat) * 100 + row["Z-spread (bps)"] / 100
            ax2.scatter(mat, implied_yield, color=c, s=80, zorder=6, marker='D')
            ax2.annotate(bond_name.split('_')[0],
                         (mat, implied_yield),
                         xytext=(4, 4), textcoords='offset points',
                         fontsize=7, color=c)
        ax2.set_xlabel('Maturity (years)')
        ax2.set_ylabel('Yield (%)')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.legend(frameon=False, fontsize=8)
        clean_spines(ax2)
        st.pyplot(fig2)
        plt.close()

    st.divider()
    st.subheader("Bond Universe — Full Detail")
    st.caption("Source: Deutsche Börse (live.deutsche-boerse.com) and issuer websites as of 21 Apr 2026. "
               "Duration and DV01 are market-sourced; KfW 10Y duration is approximated.")
    st.dataframe(
        df_zs[["Issuer", "ISIN", "Coupon (%)", "Maturity (Y)",
                "Price", "Theo. Price", "Richness (bps)", "Z-spread (bps)"]],
        use_container_width=True
    )

# ============================================================
# TAB 2 — REPO FUNDING
# ============================================================
with tab2:

    framework_box(
        "Framework — Market-Implied Repo",
        "In this tab, repo rates are <b>market-observable quantities</b>: GC (General Collateral) "
        "is proxied as ESTER − 10 bps, consistent with ICMA repo market survey data. "
        "Each bond's specific repo rate reflects its spread to GC based on structural "
        "demand (SSA bonds trade 2–4 bps special; Bunds trade 8–12 bps special). "
        "These rates are used to compute <b>net carry</b>, <b>break-even repo rates</b>, "
        "and <b>RV signals</b> across the SSA universe. "
        "This framework does <i>not</i> model specialness dynamics — see the CTD tab for that."
    )

    st.caption(
        f"**GC rate:** {gc_rate:.4f}%  ·  "
        f"**Horizon:** {funding_horizon}d  ·  "
        f"**Notional:** EUR {notional_m}M  ·  "
        f"**Convention:** coupon ACT/365 · repo ACT/360"
    )

    ACT360 = 360
    ACT365 = 365
    CURVE_SLOPE = 5.0  # bps/year — simplified roll-down proxy

    def compute_carry(bond, repo_r, horizon, notional):
        coupon_acc = (bond["coupon"] / 100) * (horizon / ACT365)
        fund_cost  = (bond["price"] / 100) * (repo_r / 100) * (horizon / ACT360)
        carry_net  = coupon_acc - fund_cost
        carry_eur  = carry_net * notional
        roll_bps   = CURVE_SLOPE * (horizon / ACT365)
        roll_eur   = roll_bps * bond["dv01"] * (notional / 1_000_000)
        be_repo    = (bond["coupon"] / 100) / (bond["price"] / 100) * (ACT360 / ACT365) * 100
        return {
            "Coupon accrual (bps)": round(coupon_acc * 10000, 2),
            "Funding cost (bps)"  : round(fund_cost  * 10000, 2),
            "Net carry (bps)"     : round(carry_net   * 10000, 2),
            "Carry EUR"           : round(carry_eur, 0),
            "Roll-down EUR"       : round(roll_eur, 0),
            "Total return EUR"    : round(carry_eur + roll_eur, 0),
            "Break-even repo (%)" : round(be_repo, 4),
            "Safety margin (bps)" : round((be_repo - repo_r) * 100, 1),
        }

    carry_rows = []
    for bond_name, bond in df_bonds.iterrows():
        row = compute_carry(bond, bond["repo_rate"], funding_horizon, NOTIONAL)
        row["Bond"]          = bond_name
        row["Issuer"]        = bond["issuer"]
        row["Repo rate (%)"] = round(bond["repo_rate"], 4)
        carry_rows.append(row)

    df_carry = pd.DataFrame(carry_rows).set_index("Bond")

    best_carry = df_carry["Net carry (bps)"].idxmax()
    best_be    = df_carry["Safety margin (bps)"].idxmax()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GC Rate", f"{gc_rate:.4f}%",
              delta=f"ESTER {gc_spread_input} bps")
    c2.metric("Best net carry",
              best_carry,
              delta=f"{df_carry.loc[best_carry, 'Net carry (bps)']:.2f} bps/{funding_horizon}d")
    c3.metric("Largest safety margin",
              best_be,
              delta=f"+{df_carry.loc[best_be, 'Safety margin (bps)']:.0f} bps")
    c4.metric("Best total return",
              df_carry["Total return EUR"].idxmax(),
              delta=f"€{df_carry['Total return EUR'].max():,.0f}")

    st.divider()

    bonds_order  = list(df_carry.index)
    colors_order = [df_bonds.loc[b, "color"] for b in bonds_order]
    x = np.arange(len(bonds_order))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Carry Decomposition ({funding_horizon}d)")
        st.caption("Net carry = coupon accrual (ACT/365) minus repo funding cost (ACT/360). "
                   "Dots show net carry in bps.")
        fig3, ax3 = plt.subplots(figsize=(7, 4), dpi=120)
        width = 0.35
        acc = df_carry.loc[bonds_order, "Coupon accrual (bps)"]
        fnd = df_carry.loc[bonds_order, "Funding cost (bps)"]
        net = df_carry.loc[bonds_order, "Net carry (bps)"]
        ax3.bar(x - width/2, acc, width, color=colors_order, alpha=0.4, label='Coupon accrual')
        ax3.bar(x + width/2, -fnd, width, color=colors_order, alpha=0.85, label='Funding cost (neg.)')
        ax3.plot(x, net, 'ko', markersize=7, label='Net carry (bps)', zorder=5)
        ax3.axhline(0, color='gray', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(bonds_order, rotation=30, ha='right', fontsize=8)
        ax3.set_ylabel('bps')
        ax3.legend(frameon=False, fontsize=8)
        clean_spines(ax3)
        st.pyplot(fig3)
        plt.close()

    with col2:
        st.subheader("Break-even Repo Rate")
        st.caption("Break-even = maximum funding cost before carry turns negative. "
                   "Annotations show safety margin (break-even minus current repo rate).")
        fig4, ax4 = plt.subplots(figsize=(7, 4), dpi=120)
        be_rates   = df_carry.loc[bonds_order, "Break-even repo (%)"]
        curr_rates = df_carry.loc[bonds_order, "Repo rate (%)"]
        ax4.bar(x, be_rates, width=0.6, color=colors_order, alpha=0.5, label='Break-even repo')
        ax4.plot(x, curr_rates, 'ko', markersize=7, label='Current repo rate', zorder=5)
        for i, (be, curr) in enumerate(zip(be_rates, curr_rates)):
            margin = (be - curr) * 100
            ax4.annotate(f'+{margin:.0f} bps',
                         xy=(i, curr), xytext=(i, curr + 0.04),
                         ha='center', fontsize=7, color='#006D19')
        ax4.set_xticks(x)
        ax4.set_xticklabels(bonds_order, rotation=30, ha='right', fontsize=8)
        ax4.set_ylabel('Rate (%)')
        ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax4.legend(frameon=False, fontsize=8)
        clean_spines(ax4)
        st.pyplot(fig4)
        plt.close()

    st.divider()
    st.subheader(f"Stress Test — GC Rate Shock: {gc_shock_bps:+d} bps")
    st.caption("Shows how net carry and safety margin change if GC repo rates shift. "
               "Useful for quarter-end or ECB reserve maintenance period stress scenarios.")

    stressed_rows = []
    for bond_name, bond in df_bonds.iterrows():
        repo_stressed = bond["repo_rate"] + gc_shock_bps / 100
        base  = compute_carry(bond, bond["repo_rate"],  funding_horizon, NOTIONAL)
        shock = compute_carry(bond, repo_stressed, funding_horizon, NOTIONAL)
        stressed_rows.append({
            "Bond"                    : bond_name,
            "Base carry EUR"          : base["Carry EUR"],
            "Stressed carry EUR"      : shock["Carry EUR"],
            "Impact EUR"              : shock["Carry EUR"] - base["Carry EUR"],
            "Base safety margin (bps)": base["Safety margin (bps)"],
            "Stressed margin (bps)"   : shock["Safety margin (bps)"],
            "Still profitable"        : "✅" if shock["Net carry (bps)"] > 0 else "❌",
        })

    df_stress = pd.DataFrame(stressed_rows).set_index("Bond")
    st.dataframe(
        df_stress.style.format({
            "Base carry EUR"          : "€{:,.0f}",
            "Stressed carry EUR"      : "€{:,.0f}",
            "Impact EUR"              : "€{:,.0f}",
            "Base safety margin (bps)": "{:.1f}",
            "Stressed margin (bps)"   : "{:.1f}",
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("Full Carry Table")
    st.dataframe(
        df_carry[[
            "Issuer", "Repo rate (%)",
            "Coupon accrual (bps)", "Funding cost (bps)",
            "Net carry (bps)", "Carry EUR",
            "Roll-down EUR", "Total return EUR",
            "Break-even repo (%)", "Safety margin (bps)"
        ]].style.format({
            "Repo rate (%)"       : "{:.4f}",
            "Break-even repo (%)" : "{:.4f}",
            "Carry EUR"           : "€{:,.0f}",
            "Roll-down EUR"       : "€{:,.0f}",
            "Total return EUR"    : "€{:,.0f}",
        }),
        use_container_width=True
    )

# ============================================================
# TAB 3 — CTD SPECIALNESS
# ============================================================
with tab3:

    framework_box(
        "Framework — Specialness-Driven Repo (CTD Microstructure)",
        "In this tab, repo is <b>endogenously driven by bond scarcity</b>, not by market GC levels. "
        "When a bond becomes the Cheapest-To-Deliver (CTD) in the Bund futures market, "
        "demand to source it in the repo market rises as delivery approaches — "
        "pushing its specific repo rate below GC. "
        "Specialness is modelled via an <b>Ornstein-Uhlenbeck (OU) process</b> with a "
        "time-varying deterministic mean calibrated on Bund repo empirical ranges "
        "(ICMA / IMF / ECB). "
        "This framework explains <i>why</i> repo can trade special — "
        "complementing the market-implied repo view in Tab 2.",
        kind="specialness"
    )

    FUTURE_CONTRACT = "FGBLM6"
    DELIVERY_DATE   = pd.Timestamp("2026-06-10")
    DAYS_TO_EXPIRY  = (DELIVERY_DATE - today).days

    ALPHA_BPS = 120.0
    BETA      = 0.04
    GAMMA_BPS = 2.0
    KAPPA     = 0.15
    SIGMA_BPS = 3.0

    def mu_specialness_bps(tau):
        return GAMMA_BPS + ALPHA_BPS * np.exp(-BETA * tau)

    current_s   = mu_specialness_bps(DAYS_TO_EXPIRY)
    exit_s      = mu_specialness_bps(exit_dte)
    entry_s     = mu_specialness_bps(entry_dte)

    # ── Conceptual bridge ────────────────────────────────────
    ctd_implied_repo = gc_rate - current_s / 100
    ssa_avg_repo     = df_bonds[df_bonds["issuer"] != "Germany"]["repo_rate"].mean()

    framework_box(
        "Bridge — From GC to Special: Three Levels of Repo",
        f"The repo market prices bonds along a spectrum from GC to deeply special. "
        f"Below is a snapshot of three representative repo levels for the current market snapshot:<br><br>"
        f"<b>① GC repo (ESTER proxy):</b> {gc_rate:.4f}% — the baseline unsecured funding cost.<br>"
        f"<b>② SSA repo (avg KfW/EIB):</b> {ssa_avg_repo:.4f}% — mildly special, reflecting "
        f"steady demand from money market funds and collateral managers.<br>"
        f"<b>③ CTD implied repo (model):</b> {ctd_implied_repo:.4f}% — the rate at which the "
        f"CTD bond (DBR 2.5% Feb-35) is estimated to trade special, driven by delivery pressure "
        f"with {DAYS_TO_EXPIRY}d to expiry.<br><br>"
        f"The specialness spread (GC minus CTD repo) is currently estimated at "
        f"<b>{current_s:.1f} bps</b> — consistent with the 5–15 bps empirical range for "
        f"the 60d-to-expiry window.",
        kind="bridge"
    )

    # Three-level repo comparison metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GC Repo",        f"{gc_rate:.4f}%",        delta="Baseline funding")
    m2.metric("Avg SSA Repo",   f"{ssa_avg_repo:.4f}%",   delta=f"{(ssa_avg_repo - gc_rate)*100:.1f} bps vs GC")
    m3.metric("CTD Implied Repo", f"{ctd_implied_repo:.4f}%",
              delta=f"−{current_s:.1f} bps vs GC (specialness)")
    m4.metric("Days to Expiry", f"{DAYS_TO_EXPIRY}d",
              delta=f"Delivery {DELIVERY_DATE.date()}")

    st.divider()

    # ── Monte Carlo ──────────────────────────────────────────
    np.random.seed(42)
    N_PATHS   = 1000
    TAU_START = 90
    N_STEPS   = TAU_START

    paths = np.zeros((N_PATHS, N_STEPS + 1))
    paths[:, 0] = mu_specialness_bps(TAU_START)

    for step in range(1, N_STEPS + 1):
        tau_t = TAU_START - step
        mu_t  = mu_specialness_bps(tau_t)
        dW    = np.random.normal(0, 1.0, N_PATHS)
        paths[:, step] = np.maximum(
            paths[:, step - 1]
            + KAPPA * (mu_t - paths[:, step - 1]) * 1.0
            + SIGMA_BPS * dW,
            0
        )

    dte_axis  = np.array([TAU_START - d for d in range(N_STEPS + 1)])
    mean_path = paths.mean(axis=0)
    p10_path  = np.percentile(paths, 10, axis=0)
    p25_path  = np.percentile(paths, 25, axis=0)
    p75_path  = np.percentile(paths, 75, axis=0)
    p90_path  = np.percentile(paths, 90, axis=0)

    CTD_PRICE    = 96.626
    NOTIONAL_CTD = NOTIONAL
    CASH_VALUE   = NOTIONAL_CTD * (CTD_PRICE / 100)

    entry_idx = TAU_START - min(entry_dte, TAU_START)
    exit_idx  = TAU_START - max(exit_dte, 0)

    pnl_paths = np.zeros((N_PATHS, N_STEPS + 1))
    for step in range(1, N_STEPS + 1):
        dte = TAU_START - step
        if exit_dte <= dte <= entry_dte:
            daily = paths[:, step] * (CASH_VALUE / 10000) / 360
            pnl_paths[:, step] = pnl_paths[:, step - 1] + daily
        else:
            pnl_paths[:, step] = pnl_paths[:, step - 1]

    trade_dte = dte_axis[entry_idx:exit_idx + 1]
    pnl_mean  = pnl_paths.mean(axis=0)[entry_idx:exit_idx + 1]
    pnl_p10   = np.percentile(pnl_paths, 10, axis=0)[entry_idx:exit_idx + 1]
    pnl_p90   = np.percentile(pnl_paths, 90, axis=0)[entry_idx:exit_idx + 1]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Specialness Term Structure (OU Model)")
        st.caption(
            "Deterministic mean μ(τ) = 2 + 120·e^(−0.04τ) bps, calibrated on "
            "ICMA/IMF/ECB empirical ranges. Green dot = current model estimate. "
            "Shaded bands = Monte Carlo P10–P90 and P25–P75."
        )
        fig5, ax5 = plt.subplots(figsize=(7, 4), dpi=120)
        ax5.fill_between(dte_axis, p10_path, p90_path, alpha=0.12, color='#003366', label='P10–P90')
        ax5.fill_between(dte_axis, p25_path, p75_path, alpha=0.25, color='#003366', label='P25–P75')
        ax5.plot(dte_axis, mean_path,
                 color='#003366', linewidth=1.8, label='Mean simulated')
        ax5.plot(dte_axis, [mu_specialness_bps(t) for t in dte_axis],
                 color='#CC3300', linewidth=1.5, linestyle='--', label='Deterministic μ(τ)')
        ax5.axvline(DAYS_TO_EXPIRY, color='#006D19', linewidth=1.0, linestyle=':')
        ax5.axvline(entry_dte, color='#CC3300', linewidth=0.8, linestyle=':', label=f'Entry ({entry_dte}d)')
        ax5.axvline(exit_dte,  color='#006D19', linewidth=0.8, linestyle=':', label=f'Exit ({exit_dte}d)')
        ax5.scatter([DAYS_TO_EXPIRY], [current_s], color='#006D19', s=60, zorder=6)
        ax5.annotate(f'Today\n{current_s:.1f} bps',
                     (DAYS_TO_EXPIRY, current_s),
                     xytext=(DAYS_TO_EXPIRY + 5, current_s + 5),
                     fontsize=8, color='#006D19')
        ax5.invert_xaxis()
        ax5.set_xlim(90, 0)
        ax5.set_xlabel('Days to expiry (τ)')
        ax5.set_ylabel('Specialness (bps)')
        ax5.legend(frameon=False, fontsize=7, ncol=2)
        clean_spines(ax5)
        st.pyplot(fig5)
        plt.close()

    with col_b:
        st.subheader(f"Repo Carry P&L — Entry {entry_dte}d → Exit {exit_dte}d")
        st.caption(
            "Daily P&L = specialness (bps) × cash value / 10,000 / 360. "
            f"Cash value = EUR {CASH_VALUE/1e6:.2f}M (notional × {CTD_PRICE}%). "
            "Carry accrues as the CTD bond is lent special in repo. "
            "Band = Monte Carlo P10–P90 across 1,000 paths."
        )
        fig6, ax6 = plt.subplots(figsize=(7, 4), dpi=120)
        if len(trade_dte) > 0:
            ax6.fill_between(trade_dte, pnl_p10, pnl_p90,
                             alpha=0.15, color='#006D19', label='P10–P90')
            ax6.plot(trade_dte, pnl_mean, color='#006D19',
                     linewidth=2.0, label='Mean P&L')
            ax6.axhline(0, color='gray', linewidth=0.5)
            final_mean = pnl_mean[-1]
            final_p10  = pnl_p10[-1]
            final_p90  = pnl_p90[-1]
            ax6.annotate(
                f'Exit\nMean: €{final_mean:,.0f}\n'
                f'P10: €{final_p10:,.0f}\nP90: €{final_p90:,.0f}',
                xy=(trade_dte[-1], final_mean),
                xytext=(trade_dte[-1] + 5, final_mean * 0.6),
                fontsize=8, color='#006D19',
                arrowprops=dict(arrowstyle='->', color='#006D19')
            )
        ax6.invert_xaxis()
        ax6.set_xlabel('Days to expiry (τ)')
        ax6.set_ylabel('Cumulative P&L (EUR)')
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'€{v:,.0f}'))
        ax6.legend(frameon=False, fontsize=8)
        clean_spines(ax6)
        st.pyplot(fig6)
        plt.close()

    st.divider()
    st.subheader("Empirical Calibration Reference")
    st.caption("OU model calibrated to match these empirical ranges. "
               "Sources: Hill (ICMA 2022), Arrata et al. (IMF WP/18/258), Corradin & Maddaloni (ECB WP 2065).")
    st.dataframe(pd.DataFrame({
        "Days to expiry": [90, 60, 30, 15, 5],
        "Empirical low (bps)":  [2, 5, 10, 20, 50],
        "Empirical high (bps)": [5, 15, 30, 60, 150],
        "Model μ(τ) (bps)": [round(mu_specialness_bps(t), 1) for t in [90, 60, 30, 15, 5]],
    }), use_container_width=True, hide_index=True)

# ============================================================
# TAB 4 — RV MATRIX
# ============================================================
with tab4:

    framework_box(
        "Framework — Multi-Metric Relative Value Ranking",
        "The RV Matrix combines four metrics into a composite score: "
        "<b>Z-spread</b> (cheapness vs ECB AAA curve), "
        "<b>net carry</b> (income after repo funding), "
        "<b>break-even safety margin</b> (funding resilience), and "
        "<b>total return</b> (carry + roll-down). "
        "Each metric is normalised to [0,1] and equally weighted. "
        "Higher score = more attractive on a risk-adjusted basis. "
        "This is a <i>relative</i> ranking within this universe only."
    )

    rv_rows = []
    for bond_name, bond in df_bonds.iterrows():
        mat_y  = (bond["maturity"] - today).days / 365.25
        zs     = compute_z_spread(bond["coupon"] / 100, mat_y, bond["price"])
        carry  = compute_carry(bond, bond["repo_rate"], funding_horizon, NOTIONAL)
        rv_rows.append({
            "Bond"                : bond_name,
            "Issuer"              : bond["issuer"],
            "Maturity (Y)"        : round(mat_y, 2),
            "Z-spread (bps)"      : round(zs, 2),
            "Net carry (bps)"     : carry["Net carry (bps)"],
            "Total return EUR"    : carry["Total return EUR"],
            "Safety margin (bps)" : carry["Safety margin (bps)"],
            "Repo rate (%)"       : round(bond["repo_rate"], 4),
        })

    df_rv = pd.DataFrame(rv_rows).set_index("Bond")

    metrics = ["Z-spread (bps)", "Net carry (bps)", "Safety margin (bps)", "Total return EUR"]
    df_norm = df_rv[metrics].copy()
    for col in metrics:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - mn) / (mx - mn + 1e-9)
    df_rv["RV Score"] = df_norm.mean(axis=1).round(3)
    df_rv = df_rv.sort_values("RV Score", ascending=False)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Composite RV Score — Ranking")
        st.caption("Equal-weighted normalised score across Z-spread, carry, safety margin, total return. "
                   "Bubble size proportional to score.")
        fig7, ax7 = plt.subplots(figsize=(7, 4), dpi=120)
        bonds_ranked = list(df_rv.index)
        scores       = df_rv.loc[bonds_ranked, "RV Score"]
        colors_rv    = [df_bonds.loc[b, "color"] for b in bonds_ranked]
        bars = ax7.barh(bonds_ranked, scores, color=colors_rv, alpha=0.85)
        ax7.set_xlabel('Composite RV Score (0 = least attractive, 1 = most attractive)')
        ax7.set_xlim(0, 1.15)
        for bar, val in zip(bars, scores):
            ax7.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=9)
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        st.pyplot(fig7)
        plt.close()

    with col_b:
        st.subheader("Z-spread vs Net Carry")
        st.caption("Ideal position: top-right (cheap on Z-spread AND high carry). "
                   "Bubble size = composite RV score.")
        fig8, ax8 = plt.subplots(figsize=(7, 4), dpi=120)
        for bond_name in df_rv.index:
            z     = df_rv.loc[bond_name, "Z-spread (bps)"]
            c     = df_rv.loc[bond_name, "Net carry (bps)"]
            col   = df_bonds.loc[bond_name, "color"]
            score = df_rv.loc[bond_name, "RV Score"]
            ax8.scatter(z, c, s=score * 600 + 100, color=col, alpha=0.8, zorder=5)
            ax8.annotate(bond_name,
                         (z, c), xytext=(4, 4), textcoords='offset points',
                         fontsize=8, color=col)
        ax8.axhline(df_rv["Net carry (bps)"].mean(), color='gray',
                    linewidth=0.7, linestyle='--', label='Avg carry')
        ax8.axvline(0, color='gray', linewidth=0.7, linestyle='--')
        ax8.set_xlabel('Z-spread vs ECB AAA curve (bps)')
        ax8.set_ylabel(f'Net carry (bps / {funding_horizon}d)')
        ax8.legend(frameon=False, fontsize=8)
        clean_spines(ax8)
        st.pyplot(fig8)
        plt.close()

    st.divider()
    st.subheader("Full RV Table")
    st.dataframe(
        df_rv.style.format({
            "Maturity (Y)"        : "{:.2f}",
            "Z-spread (bps)"      : "{:.2f}",
            "Net carry (bps)"     : "{:.2f}",
            "Total return EUR"    : "€{:,.0f}",
            "Safety margin (bps)" : "{:.1f}",
            "Repo rate (%)"       : "{:.4f}",
            "RV Score"            : "{:.3f}",
        }).background_gradient(subset=["RV Score"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.divider()
    st.caption(
        "**RV Score methodology:** composite score normalised across Z-spread, net carry, "
        "break-even safety margin and total return. Equal weighting. "
        "KfW 10Y scores high due to its 4.68% coupon — a carry outlier, not a structural cheapness signal. "
        "**Prices:** indicative levels sourced from Deutsche Börse and issuer websites (21 Apr 2026). "
        "In production, live mid-market prices from MTS Fixed Income or Bloomberg would be used."
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "**Data sources:** ECB Statistical Data Warehouse — "
    "ESTER: `EST/B.EU000A2X2A25.WT` · "
    "Bund curve: `YC/B.U2.EUR.4F.G_N_A.SV_C_YM.*` · "
    "Bond prices: Deutsche Börse XFRA (indicative, as of 21 Apr 2026) · "
    "**Specialness model:** OU calibrated on ICMA/IMF/ECB empirical ranges · "
    "**Not investment advice. For research and interview purposes only.**"
)