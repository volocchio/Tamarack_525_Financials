import itertools
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D, proj3d


@dataclass(frozen=True)
class DriverSpec:
    key: str
    label: str
    unit: str
    kind: str  # float | int | percent | thousands


def _to_internal_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) / 100.0
    if kind == "thousands":
        return float(v) * 1000.0
    return float(v)


def _to_display_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) * 100.0
    if kind == "thousands":
        return float(v) / 1000.0
    return float(v)


def _round_series_for_spec(s: pd.Series, spec: "DriverSpec") -> pd.Series:
    if spec.kind in {"int", "thousands"}:
        return s.astype(float).round(0)
    return s.astype(float).round(1)


def _disp_key(spec: "DriverSpec", v: float) -> float:
    if spec.kind in {"int", "thousands"}:
        return float(int(round(float(v))))
    return float(round(float(v), 1))


def run_model(params: Dict[str, Any]) -> Dict[str, float]:
    """Run the Tamarack 525 financial model with given parameters."""
    
    # Extract parameters
    kit_price_base = float(params["kit_price_base"])
    kit_price_escalation = float(params["kit_price_escalation"])
    base_cogs_k = float(params["base_cogs_k"])
    cogs_escalation = float(params["cogs_escalation"])
    
    units_2026 = int(params["units_2026"])
    units_2027 = int(params["units_2027"])
    units_2028 = int(params["units_2028"])
    units_2029 = int(params["units_2029"])
    units_2030 = int(params["units_2030"])
    units_2031_2035 = int(params["units_2031_2035"])
    
    eng_rate = float(params["eng_rate"])
    eng_cost_per_hour = float(params["eng_cost_per_hour"])
    eng_overhead_pct = float(params["eng_overhead_pct"])
    
    eng_hours_2026 = float(params["eng_hours_2026"])
    eng_hours_2027 = float(params["eng_hours_2027"])
    eng_hours_2028 = float(params["eng_hours_2028"])
    eng_hours_2029 = float(params["eng_hours_2029"])
    eng_hours_2030 = float(params["eng_hours_2030"])
    eng_hours_2031_2035 = float(params["eng_hours_2031_2035"])
    
    debt_amount = float(params["debt_amount"])
    debt_apr = float(params["debt_apr"])
    debt_term_years = int(params["debt_term_years"])
    tax_rate = float(params["tax_rate"])
    wacc = float(params["wacc"])
    terminal_growth = float(params["terminal_growth"])
    
    # Historical sales data
    historical_units = {
        2016: 4, 2017: 24, 2018: 52, 2019: 18, 2020: 22,
        2021: 26, 2022: 24, 2023: 16, 2024: 14, 2025: 18,
    }
    
    # Forecast units
    forecast_units = {
        2026: units_2026, 2027: units_2027, 2028: units_2028,
        2029: units_2029, 2030: units_2030,
        2031: units_2031_2035, 2032: units_2031_2035,
        2033: units_2031_2035, 2034: units_2031_2035, 2035: units_2031_2035,
    }
    
    # Engineering hours forecast
    eng_hours_forecast = {
        2026: eng_hours_2026, 2027: eng_hours_2027, 2028: eng_hours_2028,
        2029: eng_hours_2029, 2030: eng_hours_2030,
        2031: eng_hours_2031_2035, 2032: eng_hours_2031_2035,
        2033: eng_hours_2031_2035, 2034: eng_hours_2031_2035, 2035: eng_hours_2031_2035,
    }
    
    # OpEx schedule
    opex = {
        2016: 2, 2017: 3, 2018: 5, 2019: 4, 2020: 4, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 4,
        2026: 5, 2027: 6, 2028: 7, 2029: 8, 2030: 8, 2031: 8, 2032: 8, 2033: 8, 2034: 8, 2035: 8
    }
    
    years = list(range(2016, 2036))
    
    cum_cash = 0.0
    debt_balance = 0.0
    debt_drawn_total = 0.0
    quarterly_debt_payment = None
    debt_rate_annual = float(debt_apr)
    term_years = int(debt_term_years)
    
    annual_data = {}
    
    for yr in years:
        # Kit sales
        if yr in historical_units:
            units = historical_units[yr]
        elif yr in forecast_units:
            units = forecast_units[yr]
        else:
            units = 0
        
        year_idx = yr - 2016
        kit_price = (kit_price_base * 1000.0) * ((1 + kit_price_escalation) ** year_idx)
        kit_cogs = (base_cogs_k * 1000.0) * ((1 + cogs_escalation) ** year_idx)
        
        kit_revenue = units * kit_price / 1e6
        kit_cogs_total = units * kit_cogs / 1e6
        
        # Engineering services
        if yr in eng_hours_forecast:
            eng_hours = eng_hours_forecast[yr] * 1000.0
        else:
            eng_hours = 0.0
        
        eng_revenue = eng_hours * eng_rate / 1e6
        eng_direct_cost = eng_hours * eng_cost_per_hour / 1e6
        eng_overhead = eng_direct_cost * eng_overhead_pct
        eng_cogs_total = eng_direct_cost + eng_overhead
        
        # Total
        revenue = kit_revenue + eng_revenue
        cogs = kit_cogs_total + eng_cogs_total
        gross_profit = revenue - cogs
        
        opex_yr = opex.get(yr, 8.0)
        ebitda = gross_profit - opex_yr
        
        # Debt service
        if yr == 2016 and debt_amount > 0:
            if ebitda < 0:
                debt_draw = min(debt_amount, abs(ebitda))
                debt_balance = debt_draw
                debt_drawn_total = debt_draw
            else:
                debt_draw = 0.0
        else:
            debt_draw = 0.0
        
        if debt_balance > 0:
            debt_interest = debt_balance * debt_rate_annual
            if quarterly_debt_payment is None and term_years > 0:
                if debt_rate_annual > 0:
                    quarterly_debt_payment = debt_balance * debt_rate_annual / (1 - (1 + debt_rate_annual) ** (-term_years))
                else:
                    quarterly_debt_payment = debt_balance / term_years
            
            debt_payment = min(quarterly_debt_payment if quarterly_debt_payment else 0.0, debt_balance + debt_interest)
            debt_principal = max(0.0, debt_payment - debt_interest)
            debt_balance = max(0.0, debt_balance - debt_principal)
        else:
            debt_interest = 0.0
            debt_payment = 0.0
            debt_principal = 0.0
        
        taxable_income = ebitda - debt_interest
        taxes = max(0.0, taxable_income) * tax_rate
        
        fcf_after_tax = ebitda - taxes
        net_cash_after_debt = fcf_after_tax + debt_draw - debt_payment
        cum_cash += net_cash_after_debt
        
        annual_data[yr] = {
            "EBITDA": float(ebitda),
            "Taxes": float(taxes),
        }
    
    df = pd.DataFrame.from_dict(annual_data, orient="index")
    df.index.name = "Year"
    
    unlevered_taxes = (df["EBITDA"].clip(lower=0.0) * float(tax_rate)).astype(float)
    unlevered_fcf = (df["EBITDA"].astype(float) - unlevered_taxes).astype(float)
    
    discount_year0 = int(df.index.min())
    discount_t = (df.index - discount_year0 + 1).astype(int)
    discount_factor = pd.Series((1 / (1 + float(wacc)) ** discount_t).astype(float), index=df.index)
    
    pv_fcf = (unlevered_fcf * discount_factor).astype(float)
    
    tv = np.nan
    pv_tv = np.nan
    if float(wacc) > float(terminal_growth):
        tv = float(unlevered_fcf.iloc[-1]) * (1 + float(terminal_growth)) / (float(wacc) - float(terminal_growth))
        pv_tv = tv * float(discount_factor.iloc[-1])
    
    pv_explicit = float(pv_fcf.sum())
    enterprise_value = pv_explicit + (float(pv_tv) if not np.isnan(pv_tv) else 0.0)
    
    return {
        "Enterprise Value ($M)": float(enterprise_value),
        "PV Explicit FCF ($M)": float(pv_explicit),
        "PV Terminal Value ($M)": float(0.0 if np.isnan(pv_tv) else float(pv_tv)),
        "Ending Cash ($M)": float(cum_cash),
        "Total EBITDA ($M)": float(df["EBITDA"].sum()),
    }


def build_baseline_params() -> Dict[str, Any]:
    """Build baseline parameters from sidebar inputs."""
    st.sidebar.header("Baseline Inputs")
    
    st.sidebar.header("Kit Sales")
    kit_price_base = st.sidebar.slider("Base Kit Price (2016, $k)", min_value=100, max_value=2000, value=800, step=50)
    kit_price_escalation = st.sidebar.slider("Annual Price Escalation (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100
    base_cogs_k = st.sidebar.slider("Base COGS per Kit (2016, $k)", min_value=50, max_value=1000, value=400, step=25)
    cogs_escalation = st.sidebar.slider("Annual COGS Escalation (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5) / 100
    
    st.sidebar.header("Kit Sales Forecast (2026-2035)")
    units_2026 = st.sidebar.slider("2026 Units", min_value=0, max_value=100, value=20, step=5)
    units_2027 = st.sidebar.slider("2027 Units", min_value=0, max_value=100, value=25, step=5)
    units_2028 = st.sidebar.slider("2028 Units", min_value=0, max_value=100, value=30, step=5)
    units_2029 = st.sidebar.slider("2029 Units", min_value=0, max_value=100, value=35, step=5)
    units_2030 = st.sidebar.slider("2030 Units", min_value=0, max_value=100, value=40, step=5)
    units_2031_2035 = st.sidebar.slider("2031-2035 Units (annual)", min_value=0, max_value=100, value=40, step=5)
    
    st.sidebar.header("Engineering Services")
    eng_rate = st.sidebar.slider("Billing Rate ($/hr)", min_value=50, max_value=500, value=200, step=10)
    eng_cost_per_hour = st.sidebar.slider("Cost per Billable Hour ($/hr)", min_value=20, max_value=300, value=120, step=10)
    eng_overhead_pct = st.sidebar.slider("Engineering Overhead (%)", min_value=0.0, max_value=100.0, value=20.0, step=5.0) / 100
    
    st.sidebar.header("Engineering Hours Forecast (2026-2035)")
    eng_hours_2026 = st.sidebar.slider("2026 Hours (k)", min_value=0, max_value=50, value=5, step=1)
    eng_hours_2027 = st.sidebar.slider("2027 Hours (k)", min_value=0, max_value=50, value=8, step=1)
    eng_hours_2028 = st.sidebar.slider("2028 Hours (k)", min_value=0, max_value=50, value=10, step=1)
    eng_hours_2029 = st.sidebar.slider("2029 Hours (k)", min_value=0, max_value=50, value=12, step=1)
    eng_hours_2030 = st.sidebar.slider("2030 Hours (k)", min_value=0, max_value=50, value=15, step=1)
    eng_hours_2031_2035 = st.sidebar.slider("2031-2035 Hours (k, annual)", min_value=0, max_value=50, value=15, step=1)
    
    st.sidebar.header("Financial")
    debt_amount = st.sidebar.slider("Max Debt Available ($M)", min_value=0.0, max_value=100.0, value=20.0, step=5.0)
    debt_apr = st.sidebar.slider("Debt APR (%)", min_value=0.0, max_value=20.0, value=8.0, step=0.5) / 100
    debt_term_years = st.sidebar.slider("Debt Term (Years)", min_value=1, max_value=15, value=5, step=1)
    tax_rate = st.sidebar.slider("Income Tax Rate (%)", min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
    wacc = st.sidebar.slider("WACC (%)", min_value=0.0, max_value=30.0, value=12.0, step=0.5) / 100
    terminal_growth = st.sidebar.slider("Terminal Growth Rate (%)", min_value=-2.0, max_value=8.0, value=2.5, step=0.5) / 100
    
    return {
        "kit_price_base": float(kit_price_base),
        "kit_price_escalation": float(kit_price_escalation),
        "base_cogs_k": float(base_cogs_k),
        "cogs_escalation": float(cogs_escalation),
        "units_2026": int(units_2026),
        "units_2027": int(units_2027),
        "units_2028": int(units_2028),
        "units_2029": int(units_2029),
        "units_2030": int(units_2030),
        "units_2031_2035": int(units_2031_2035),
        "eng_rate": float(eng_rate),
        "eng_cost_per_hour": float(eng_cost_per_hour),
        "eng_overhead_pct": float(eng_overhead_pct),
        "eng_hours_2026": float(eng_hours_2026),
        "eng_hours_2027": float(eng_hours_2027),
        "eng_hours_2028": float(eng_hours_2028),
        "eng_hours_2029": float(eng_hours_2029),
        "eng_hours_2030": float(eng_hours_2030),
        "eng_hours_2031_2035": float(eng_hours_2031_2035),
        "debt_amount": float(debt_amount),
        "debt_apr": float(debt_apr),
        "debt_term_years": int(debt_term_years),
        "tax_rate": float(tax_rate),
        "wacc": float(wacc),
        "terminal_growth": float(terminal_growth),
    }


def build_driver_catalog(baseline: Dict[str, Any]) -> List[DriverSpec]:
    """Build catalog of available sensitivity drivers."""
    return [
        DriverSpec("kit_price_base", "Base Kit Price (2016)", "$k/kit", "int"),
        DriverSpec("kit_price_escalation", "Kit Price Escalation", "%", "percent"),
        DriverSpec("base_cogs_k", "Base COGS per Kit (2016)", "$k/kit", "int"),
        DriverSpec("cogs_escalation", "COGS Escalation", "%", "percent"),
        DriverSpec("units_2026", "2026 Units", "units", "int"),
        DriverSpec("units_2027", "2027 Units", "units", "int"),
        DriverSpec("units_2028", "2028 Units", "units", "int"),
        DriverSpec("units_2029", "2029 Units", "units", "int"),
        DriverSpec("units_2030", "2030 Units", "units", "int"),
        DriverSpec("units_2031_2035", "2031-2035 Units (annual)", "units", "int"),
        DriverSpec("eng_rate", "Engineering Billing Rate", "$/hr", "int"),
        DriverSpec("eng_cost_per_hour", "Engineering Cost per Hour", "$/hr", "int"),
        DriverSpec("eng_overhead_pct", "Engineering Overhead", "%", "percent"),
        DriverSpec("eng_hours_2026", "2026 Eng Hours", "k hrs", "int"),
        DriverSpec("eng_hours_2027", "2027 Eng Hours", "k hrs", "int"),
        DriverSpec("eng_hours_2028", "2028 Eng Hours", "k hrs", "int"),
        DriverSpec("eng_hours_2029", "2029 Eng Hours", "k hrs", "int"),
        DriverSpec("eng_hours_2030", "2030 Eng Hours", "k hrs", "int"),
        DriverSpec("eng_hours_2031_2035", "2031-2035 Eng Hours (annual)", "k hrs", "int"),
        DriverSpec("wacc", "WACC", "%", "percent"),
        DriverSpec("debt_amount", "Max Debt Available", "$M", "float"),
        DriverSpec("debt_apr", "Debt APR", "%", "percent"),
        DriverSpec("debt_term_years", "Debt Term", "years", "int"),
        DriverSpec("tax_rate", "Income Tax Rate", "%", "percent"),
        DriverSpec("terminal_growth", "Terminal Growth Rate", "%", "percent"),
    ]


def _grid_values(spec: DriverSpec, low_disp: float, high_disp: float, points: int) -> List[float]:
    if points < 2:
        points = 2
    
    if spec.kind in {"int", "thousands"}:
        vals = np.linspace(float(low_disp), float(high_disp), int(points))
        vals = np.unique(np.round(vals).astype(int)).tolist()
        return [float(v) for v in vals]
    
    vals = np.linspace(float(low_disp), float(high_disp), int(points)).tolist()
    return [float(v) for v in vals]


def _apply_driver_value(params: Dict[str, Any], spec: DriverSpec, disp_value: float) -> None:
    internal = _to_internal_value(spec.kind, disp_value)
    params[spec.key] = internal


def _driver_disp_bounds(spec: DriverSpec) -> Tuple[float | None, float | None]:
    bounds_by_key: Dict[str, Tuple[float, float]] = {
        "kit_price_base": (100.0, 2000.0),
        "kit_price_escalation": (0.0, 10.0),
        "base_cogs_k": (50.0, 1000.0),
        "cogs_escalation": (0.0, 10.0),
        "units_2026": (0.0, 100.0),
        "units_2027": (0.0, 100.0),
        "units_2028": (0.0, 100.0),
        "units_2029": (0.0, 100.0),
        "units_2030": (0.0, 100.0),
        "units_2031_2035": (0.0, 100.0),
        "eng_rate": (50.0, 500.0),
        "eng_cost_per_hour": (20.0, 300.0),
        "eng_overhead_pct": (0.0, 100.0),
        "eng_hours_2026": (0.0, 50.0),
        "eng_hours_2027": (0.0, 50.0),
        "eng_hours_2028": (0.0, 50.0),
        "eng_hours_2029": (0.0, 50.0),
        "eng_hours_2030": (0.0, 50.0),
        "eng_hours_2031_2035": (0.0, 50.0),
        "wacc": (0.0, 30.0),
        "debt_amount": (0.0, 100.0),
        "debt_apr": (0.0, 20.0),
        "debt_term_years": (1.0, 15.0),
        "tax_rate": (0.0, 40.0),
        "terminal_growth": (-2.0, 8.0),
    }
    
    if spec.key not in bounds_by_key:
        return (None, None)
    return bounds_by_key[spec.key]


def _default_range_for_driver(spec: DriverSpec, baseline_disp: float) -> Tuple[float, float]:
    lo_bound, hi_bound = _driver_disp_bounds(spec)
    
    if spec.kind == "percent":
        span = max(2.0, abs(float(baseline_disp)) * 0.25)
        low = float(baseline_disp) - float(span)
        high = float(baseline_disp) + float(span)
    elif spec.kind in {"int", "thousands"}:
        span = max(1.0, abs(float(baseline_disp)) * 0.25)
        low = float(baseline_disp) - float(span)
        high = float(baseline_disp) + float(span)
    else:
        span = max(0.1, abs(float(baseline_disp)) * 0.25)
        low = float(baseline_disp) - float(span)
        high = float(baseline_disp) + float(span)
    
    if lo_bound is not None:
        low = max(float(lo_bound), float(low))
        high = max(float(lo_bound), float(high))
    if hi_bound is not None:
        low = min(float(hi_bound), float(low))
        high = min(float(hi_bound), float(high))
    
    if float(high) < float(low):
        low, high = float(high), float(low)
    
    if float(high) == float(low):
        if hi_bound is not None:
            high = min(float(hi_bound), float(low) + 1.0)
        else:
            high = float(low) + 1.0
    
    return (float(low), float(high))


def run_sensitivity(
    baseline: Dict[str, Any],
    d1: DriverSpec,
    d2: DriverSpec,
    d3: DriverSpec,
    d1_vals: List[float],
    d2_vals: List[float],
    d3_vals: List[float],
    metric: str,
) -> pd.DataFrame:
    rows = []
    
    for v3, v1, v2 in itertools.product(d3_vals, d1_vals, d2_vals):
        params = dict(baseline)
        _apply_driver_value(params, d1, v1)
        _apply_driver_value(params, d2, v2)
        _apply_driver_value(params, d3, v3)
        
        outputs = run_model(params)
        
        rows.append(
            {
                d1.label: v1,
                d2.label: v2,
                d3.label: v3,
                "Metric": metric,
                "Value": float(outputs[metric]),
            }
        )
    
    df = pd.DataFrame(rows)
    for spec in (d1, d2, d3):
        col = spec.label
        if col in df.columns:
            if spec.kind in {"int", "thousands"}:
                df[col] = df[col].round(0).astype(int)
            else:
                df[col] = df[col].astype(float).round(2)
    if "Value" in df.columns:
        df["Value"] = df["Value"].astype(float).round(1)
    return df


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    x_tick_fmt: str = "%.1f",
    y_tick_fmt: str = "%.1f",
    fmt: str = "%.0f",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    z = np.ma.masked_invalid(pivot.values.astype(float))
    base_cmap = plt.get_cmap("Greens")
    cmap = LinearSegmentedColormap.from_list(
        "light_to_dark_green",
        base_cmap(np.linspace(0.25, 0.95, 256)),
    )
    cmap.set_bad(color="#EEEEEE")
    vals = pivot.values.astype(float)
    if np.isfinite(vals).any():
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if float(vmin) == float(vmax):
            vmax = float(vmin) + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([x_tick_fmt % float(v) for v in pivot.columns.tolist()], rotation=45, ha="right")
    
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([y_tick_fmt % float(v) for v in pivot.index.tolist()])
    
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = float(pivot.iloc[i, j])
            if not np.isfinite(v):
                continue
            r, g, b, _ = cmap(norm(v))
            luminance = 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)
            txt_color = "black" if luminance > 0.62 else "white"
            ax.text(j, i, (fmt % v), ha="center", va="center", fontsize=7, color=txt_color)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    
    fig.tight_layout()
    return fig


def render_sensitivity_app(
    baseline_params: Dict[str, Any] | None = None,
    show_title: bool = True,
) -> None:
    if show_title:
        st.title("Tamarack 525 – 3-Driver Sensitivity Study")
    
    if baseline_params is None:
        baseline_params = build_baseline_params()
    
    st.header("Baseline Outputs")
    baseline_outputs = run_model(baseline_params)
    baseline_out_df = pd.DataFrame({
        "Metric": list(baseline_outputs.keys()),
        "Baseline": list(baseline_outputs.values()),
    })
    baseline_out_df["Baseline"] = baseline_out_df["Baseline"].astype(float).round(1)
    st.dataframe(baseline_out_df, hide_index=True, use_container_width=False)
    
    st.header("Sensitivity Study")
    
    drivers = build_driver_catalog(baseline_params)
    driver_by_key = {d.key: d for d in drivers}
    driver_category = {
        "kit_price_base": "Kit Sales",
        "kit_price_escalation": "Kit Sales",
        "base_cogs_k": "Kit Sales",
        "cogs_escalation": "Kit Sales",
        "units_2026": "Kit Sales",
        "units_2027": "Kit Sales",
        "units_2028": "Kit Sales",
        "units_2029": "Kit Sales",
        "units_2030": "Kit Sales",
        "units_2031_2035": "Kit Sales",
        "eng_rate": "Engineering",
        "eng_cost_per_hour": "Engineering",
        "eng_overhead_pct": "Engineering",
        "eng_hours_2026": "Engineering",
        "eng_hours_2027": "Engineering",
        "eng_hours_2028": "Engineering",
        "eng_hours_2029": "Engineering",
        "eng_hours_2030": "Engineering",
        "eng_hours_2031_2035": "Engineering",
        "wacc": "Financial",
        "debt_amount": "Financial",
        "debt_apr": "Financial",
        "debt_term_years": "Financial",
        "tax_rate": "Financial",
        "terminal_growth": "Financial",
    }
    driver_labels = {d.key: f"{driver_category.get(d.key, 'Other')} - {d.label}" for d in drivers}
    
    def _key_index(keys: List[str], desired: str, fallback: int = 0) -> int:
        return keys.index(desired) if desired in keys else fallback
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 1</div>',
            unsafe_allow_html=True,
        )
        d1_options = [d.key for d in drivers]
        d1_default = "units_2030"
        d1_key = st.selectbox(
            "Driver 1",
            options=d1_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d1_options, d1_default, 0),
            label_visibility="collapsed",
        )
    with col_b:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 2</div>',
            unsafe_allow_html=True,
        )
        d2_options = [d.key for d in drivers if d.key != d1_key]
        d2_default = "kit_price_base" if "kit_price_base" in d2_options else (d2_options[0] if len(d2_options) > 0 else d1_key)
        d2_key = st.selectbox(
            "Driver 2",
            options=d2_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d2_options, d2_default, 0),
            label_visibility="collapsed",
        )
    with col_c:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 3 (Heatmap Slicer)</div>',
            unsafe_allow_html=True,
        )
        d3_options = [d.key for d in drivers if d.key not in {d1_key, d2_key}]
        d3_default = "eng_hours_2030" if "eng_hours_2030" in d3_options else (d3_options[0] if len(d3_options) > 0 else d1_key)
        d3_key = st.selectbox(
            "Driver 3",
            options=d3_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d3_options, d3_default, 0),
            label_visibility="collapsed",
        )
    
    d1 = driver_by_key[d1_key]
    d2 = driver_by_key[d2_key]
    d3 = driver_by_key[d3_key]
    
    metrics = list(baseline_outputs.keys())
    metric_labels = {m: f"Financial - {m}" for m in metrics}
    metric = st.selectbox("Metric to Sensitize", options=metrics, format_func=lambda m: metric_labels.get(m, m), index=0)
    
    b1 = _to_display_value(d1.kind, float(baseline_params[d1.key]))
    b2 = _to_display_value(d2.kind, float(baseline_params[d2.key]))
    b3 = _to_display_value(d3.kind, float(baseline_params[d3.key]))
    
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 1 Range</div>',
            unsafe_allow_html=True,
        )
        d1_low_default, d1_high_default = _default_range_for_driver(d1, float(b1))
        d1_min, d1_max = _driver_disp_bounds(d1)
        if d1_min is None or d1_max is None:
            d1_low = st.number_input("Low", value=float(d1_low_default), key=f"d1_low__{d1.key}")
            d1_high = st.number_input("High", value=float(d1_high_default), key=f"d1_high__{d1.key}")
        else:
            d1_low = st.number_input("Low", min_value=float(d1_min), max_value=float(d1_max), value=float(d1_low_default), key=f"d1_low__{d1.key}")
            d1_high = st.number_input("High", min_value=float(d1_min), max_value=float(d1_max), value=float(d1_high_default), key=f"d1_high__{d1.key}")
        d1_points = st.number_input("Points", min_value=2, max_value=25, value=5, step=1, key=f"d1_points__{d1.key}")
    with cfg2:
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 2 Range</div>',
            unsafe_allow_html=True,
        )
        d2_low_default, d2_high_default = _default_range_for_driver(d2, float(b2))
        d2_min, d2_max = _driver_disp_bounds(d2)
        if d2_min is None or d2_max is None:
            d2_low = st.number_input("Low ", value=float(d2_low_default), key=f"d2_low__{d2.key}")
            d2_high = st.number_input("High ", value=float(d2_high_default), key=f"d2_high__{d2.key}")
        else:
            d2_low = st.number_input("Low ", min_value=float(d2_min), max_value=float(d2_max), value=float(d2_low_default), key=f"d2_low__{d2.key}")
            d2_high = st.number_input("High ", min_value=float(d2_min), max_value=float(d2_max), value=float(d2_high_default), key=f"d2_high__{d2.key}")
        d2_points = st.number_input("Points ", min_value=2, max_value=25, value=5, step=1, key=f"d2_points__{d2.key}")
    with cfg3:
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 3 Range</div>',
            unsafe_allow_html=True,
        )
        d3_low_default, d3_high_default = _default_range_for_driver(d3, float(b3))
        d3_min, d3_max = _driver_disp_bounds(d3)
        if d3_min is None or d3_max is None:
            d3_low = st.number_input("Low  ", value=float(d3_low_default), key=f"d3_low__{d3.key}")
            d3_high = st.number_input("High  ", value=float(d3_high_default), key=f"d3_high__{d3.key}")
        else:
            d3_low = st.number_input("Low  ", min_value=float(d3_min), max_value=float(d3_max), value=float(d3_low_default), key=f"d3_low__{d3.key}")
            d3_high = st.number_input("High  ", min_value=float(d3_min), max_value=float(d3_max), value=float(d3_high_default), key=f"d3_high__{d3.key}")
        d3_points = st.number_input("Points  ", min_value=2, max_value=25, value=3, step=1, key=f"d3_points__{d3.key}")
    
    d1_vals = _grid_values(d1, float(d1_low), float(d1_high), int(d1_points))
    d2_vals = _grid_values(d2, float(d2_low), float(d2_high), int(d2_points))
    d3_vals = _grid_values(d3, float(d3_low), float(d3_high), int(d3_points))
    
    def _fmt_value(spec: DriverSpec, v: float) -> str:
        if spec.kind in {"int", "thousands"}:
            return f"{int(round(float(v)))}"
        return f"{float(v):.1f}"
    
    def _tick_fmt(spec: DriverSpec) -> str:
        if spec.kind in {"int", "thousands"}:
            return "%.0f"
        return "%.1f"
    
    scenario_count = len(d1_vals) * len(d2_vals) * len(d3_vals)
    st.write(f"Scenarios: {scenario_count}")
    
    if scenario_count > 5000:
        st.error("Too many scenarios. Reduce points (target <= 5,000).")
        return
    
    sig = (
        str(d1.key),
        str(d2.key),
        str(d3.key),
        str(metric),
        float(d1_low),
        float(d1_high),
        int(d1_points),
        float(d2_low),
        float(d2_high),
        int(d2_points),
        float(d3_low),
        float(d3_high),
        int(d3_points),
    )
    
    if "sensitivity_results" not in st.session_state:
        st.session_state.sensitivity_results = None
        st.session_state.sensitivity_sig = None
    
    if st.session_state.sensitivity_sig is not None and st.session_state.sensitivity_sig != sig:
        st.session_state.sensitivity_results = None
        st.session_state.sensitivity_sig = None
    
    run = st.button("Run Sensitivity Study")
    if run:
        st.session_state.sensitivity_results = run_sensitivity(baseline_params, d1, d2, d3, d1_vals, d2_vals, d3_vals, metric)
        st.session_state.sensitivity_sig = sig
    
    results = st.session_state.sensitivity_results
    if results is None:
        st.info("Click 'Run Sensitivity Study' to generate results.")
        return
    
    with st.expander("Scenario Results (Long Form)", expanded=False):
        st.dataframe(results, use_container_width=True)
    
    st.subheader("Heatmap Slices")
    st.markdown(
        f"**Driver 1 (X-axis):** {d1.label}  \n"
        f"**Driver 2 (Y-axis):** {d2.label}  \n"
        f"**Driver 3 (Tabs/Slices):** {d3.label}"
    )
    
    d3_disp_vals = [_disp_key(d3, float(v)) for v in d3_vals]
    seen = set()
    d3_disp_vals = [v for v in d3_disp_vals if not (v in seen or seen.add(v))]
    
    tabs = st.tabs([f"{d3.label} = {_fmt_value(d3, float(v))}" for v in d3_disp_vals])
    for tab, v3 in zip(tabs, d3_disp_vals):
        with tab:
            st.markdown(f"**Slice:** {d3.label} = {_fmt_value(d3, float(v3))}")
            slice_df = results[_round_series_for_spec(results[d3.label], d3) == float(v3)]
            pivot = slice_df.pivot(index=d2.label, columns=d1.label, values="Value").sort_index().sort_index(axis=1).astype(float).round(0)
            st.dataframe(pivot, use_container_width=True)
            fig = plot_heatmap(
                pivot,
                x_label=d1.label,
                y_label=d2.label,
                title=f"{metric} | {d3.label}={_fmt_value(d3, float(v3))}",
                x_tick_fmt=_tick_fmt(d1),
                y_tick_fmt=_tick_fmt(d2),
            )
            st.pyplot(fig)
            plt.close(fig)
