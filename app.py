import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from pathlib import Path
from typing import Dict

from app_sense import render_sensitivity_app

st.set_page_config(layout='wide')

_asset_dir = Path(__file__).resolve().parent
_img_a320 = _asset_dir / 'tamarack_CJ1+.jpg'
_img_logo = _asset_dir / 'logo.png'

col_img_1, col_img_2, _ = st.columns([2, 2, 4])
with col_img_1:
    if _img_a320.exists():
        st.image(str(_img_a320), width=340)
with col_img_2:
    if _img_logo.exists():
        st.image(str(_img_logo), width=500)

st.title('Tamarack 525 Financial Model')

st.markdown(
    """
<style>
div[data-testid="stRadio"] {
  background: #F7FAFF;
  border: 2px solid #3B82F6;
  border-radius: 12px;
  padding: 12px 14px;
  margin: 6px 0 14px 0;
}
div[data-testid="stRadio"] label p {
  font-weight: 700 !important;
  font-size: 1.05rem !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 10px;
  flex-wrap: nowrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label {
  background: #FFFFFF;
  border: 1px solid #BFDBFE;
  border-radius: 10px;
  padding: 8px 10px;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
  border-color: #1D4ED8;
  background: #DBEAFE;
  box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18);
}

div[data-testid="stMetric"] {
  background: #ECFDF5;
  border: 3px solid #10B981;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 12px 22px rgba(16, 185, 129, 0.18);
}
div[data-testid="stMetric"] label p {
  font-weight: 900 !important;
}
div[data-testid="stMetric"] label {
  font-weight: 900 !important;
  font-size: 1.05rem !important;
  color: #065F46 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-weight: 900 !important;
  font-size: 2.4rem !important;
  color: #064E3B !important;
}

/* Wrap dataframe column headers */
div[data-testid="stDataFrame"] th {
  white-space: normal !important;
  word-wrap: break-word !important;
  max-width: 120px !important;
  min-width: 80px !important;
  font-size: 0.85rem !important;
  line-height: 1.2 !important;
  padding: 8px 4px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

col_main, col_right = st.columns([0.6, 3.4])
with col_right:
    def _mode_label(v: str) -> str:
        return 'Standalone' if str(v).startswith('Standalone') else 'Sensitivity'

    col_mode, col_ev = st.columns([2.0, 1.3], gap="small")
    with col_mode:
        mode = st.radio(
            'Mode',
            options=['Standalone Model', 'Sensitivity Study (3 Drivers, 1 Output)'],
            horizontal=True,
            format_func=_mode_label,
        )
    with col_ev:
        ev_placeholder = st.empty()
        ev_placeholder.metric(label="Enterprise Value ($M)", value="—")
        st.caption("*Operated under IP license agreement for winglets and Starlink*")
with col_main:
    pass

if mode == 'Sensitivity Study (3 Drivers, 1 Output)':
    render_sensitivity_app(baseline_params=None, show_title=False)
    st.stop()

# Sidebar inputs
st.sidebar.header('Project Selection')
enable_525 = st.sidebar.checkbox('Enable 525 Winglet Project', value=True)
enable_510 = st.sidebar.checkbox('Enable 510 Mustang Winglet Project', value=True)
enable_604 = st.sidebar.checkbox('Enable Challenger 604 Family Project', value=True)
enable_engineering = st.sidebar.checkbox('Enable Engineering Services', value=True)
enable_starlink = st.sidebar.checkbox('Enable Starlink STC Project', value=True)

st.sidebar.header('525 Winglet Sales Revenue')
kit_price_base = st.sidebar.slider('Base 525 Winglet Price (2026, $k)', min_value=100, max_value=2000, value=250, step=50)
kit_price_escalation = st.sidebar.slider('Annual Price Escalation (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100
base_cogs_k = st.sidebar.slider('Base COGS per 525 Winglet (2026, $k)', min_value=50, max_value=1000, value=125, step=25)
cogs_escalation = st.sidebar.slider('Annual COGS Escalation (%)', min_value=0.0, max_value=10.0, value=3.0, step=0.5) / 100

st.sidebar.header('525 Winglet Sales Forecast (2026-2035)')
st.sidebar.markdown('**TAM: 2,200 existing + 60 new/year (excludes CJ4)**')
units_525_2026 = st.sidebar.slider('2026 Target 525 Sales (units)', min_value=0, max_value=100, value=20, step=1)
penetration_525_2035 = st.sidebar.slider('2035 Target 525 Penetration (%)', min_value=0, max_value=100, value=50, step=5)

st.sidebar.header('510 Mustang Winglet Sales Revenue')
st.sidebar.markdown('**TAM: 450 aircraft (no longer produced)**')
kit_510_price_base = st.sidebar.slider('Base 510 Winglet Price (2028, $k)', min_value=100, max_value=500, value=225, step=25)
kit_510_cogs = st.sidebar.slider('510 Winglet COGS ($k)', min_value=50, max_value=300, value=100, step=10)

st.sidebar.header('510 Mustang Winglet Forecast (2028-2035)')
units_510_2028 = st.sidebar.slider('2028 Target 510 Sales (units)', min_value=0, max_value=50, value=30, step=1)
penetration_510_2035 = st.sidebar.slider('2035 Target 510 Penetration (%)', min_value=0, max_value=100, value=50, step=5)

st.sidebar.header('Engineering Services Revenue')
eng_rate = st.sidebar.slider('Billing Rate ($/hr)', min_value=50, max_value=500, value=200, step=10)
eng_cost_per_hour = st.sidebar.slider('Cost per Billable Hour ($/hr)', min_value=20, max_value=300, value=120, step=10)
eng_overhead_pct = st.sidebar.slider('Engineering Overhead (%)', min_value=0.0, max_value=100.0, value=20.0, step=5.0) / 100

st.sidebar.header('Engineering Hours Forecast (2026-2035)')
eng_hours_start = st.sidebar.slider('2026 Starting Hours (k)', min_value=0, max_value=50, value=5, step=1)
eng_hours_growth = st.sidebar.slider('Annual Hours Growth (k/year)', min_value=-10, max_value=10, value=2, step=1)

st.sidebar.header('Starlink STC Revenue')
starlink_price_base = st.sidebar.slider('Base Starlink Price (2026, $k)', min_value=50, max_value=300, value=149, step=1)
starlink_price_escalation = st.sidebar.slider('Annual Starlink Price Escalation (%)', min_value=-30.0, max_value=30.0, value=3.0, step=0.5) / 100
starlink_cogs = st.sidebar.slider('Starlink COGS per Unit ($k)', min_value=0, max_value=10, value=2, step=1)
starlink_install = st.sidebar.slider('Starlink Install Cost per Unit ($k)', min_value=0, max_value=20, value=4, step=1)
starlink_eng_commitment = st.sidebar.slider('Starlink Engineering Commitment ($k)', min_value=0, max_value=500, value=200, step=10)

st.sidebar.header('Starlink STC Forecast (2026-2035)')
st.sidebar.markdown('**TAM: 2,650 aircraft (525 + 510 series)**')
units_starlink_2026 = st.sidebar.slider('2026 Target Starlink Sales (units)', min_value=0, max_value=100, value=10, step=1)
penetration_starlink_2035 = st.sidebar.slider('2035 Target Starlink Penetration (%)', min_value=0, max_value=100, value=50, step=5)

st.sidebar.header('Challenger 604 Family Winglet Sales Revenue')
st.sidebar.markdown('**TAM: 1,200 existing (as of 2025) + 35 new/year**')
kit_604_price_base = st.sidebar.slider('Base 604 Winglet Price (2027, $k)', min_value=100, max_value=1000, value=600, step=50)
kit_604_price_escalation = st.sidebar.slider('Annual 604 Price Escalation (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100
kit_604_cogs = st.sidebar.slider('604 Winglet COGS ($k)', min_value=50, max_value=500, value=150, step=10)
kit_604_cert_cost = st.sidebar.slider('604 Certification Cost ($M)', min_value=0, max_value=30, value=16, step=1)

st.sidebar.header('Challenger 604 Family Forecast (2027-2035)')
units_604_2027 = st.sidebar.slider('2027 Target 604 Sales (units)', min_value=0, max_value=100, value=15, step=1)
penetration_604_2035 = st.sidebar.slider('2035 Target 604 Penetration (%)', min_value=0, max_value=100, value=40, step=5)

st.sidebar.header('Financial')
opex_growth_rate = st.sidebar.slider('Annual OpEx Growth Rate (%)', min_value=0.0, max_value=10.0, value=3.0, step=0.5) / 100
debt_amount = st.sidebar.slider('Max Debt Available ($M)', min_value=0.0, max_value=100.0, value=20.0, step=5.0)
debt_apr = st.sidebar.slider('Debt APR (%)', min_value=0.0, max_value=20.0, value=8.0, step=0.5) / 100
debt_term_years = st.sidebar.slider('Debt Term (Years)', min_value=1, max_value=15, value=5, step=1)
tax_rate = st.sidebar.slider('Income Tax Rate (%)', min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
wacc = st.sidebar.slider('WACC (%)', min_value=0.0, max_value=30.0, value=12.0, step=0.5) / 100
terminal_growth = st.sidebar.slider('Terminal Growth Rate (%)', min_value=-2.0, max_value=8.0, value=2.5, step=0.5) / 100

# Historical sales data (2016-2025)
historical_units = {
    2016: 4,
    2017: 24,
    2018: 52,
    2019: 18,
    2020: 22,
    2021: 26,
    2022: 24,
    2023: 16,
    2024: 14,
    2025: 18,
}

# TAM constants
base_tam_525 = 2200  # Existing M2/CJ1/CJ2/CJ3/CJ3+ aircraft
base_tam_510 = 450
base_tam_604 = 1200  # Existing Challenger 604/605/650 aircraft as of 2025
base_tam_starlink = 2650  # All 525 series + 510
new_525_per_year = 60  # New M2/CJ1/CJ2/CJ3/CJ3+ per year (winglet eligible)
new_cj4_per_year = 31  # New CJ4 (525C) per year - Starlink only
new_604_per_year = 35  # New Challenger 650 per year (same family as 604)

# 604 Certification project (2-year project: 2026-2027)
# $16M total certification cost split evenly over 2 years

# 525 winglet forecast (2026-2035) - Annual sales ramping to achieve cumulative penetration target
# Start with 218 already installed, add units_525_2026 in first year, ramp to hit penetration target by 2035
forecast_units = {}
cum_units_baseline = 218  # Already installed before 2016
cum_units_historical = sum(historical_units.values())  # Historical sales 2016-2025

for i, yr in enumerate(range(2026, 2036)):
    years_elapsed = yr - 2026
    total_years = 9  # 2026 to 2035
    
    # Calculate target cumulative units for this year based on penetration ramp
    tam_yr = base_tam_525 + (new_525_per_year * (yr - 2016))
    tam_2035 = base_tam_525 + (new_525_per_year * (2035 - 2016))
    
    # Penetration ramps from current (218+historical) to target by 2035
    current_penetration = (cum_units_baseline + cum_units_historical) / (base_tam_525 + (new_525_per_year * (2025 - 2016))) * 100
    target_penetration_yr = current_penetration + (penetration_525_2035 - current_penetration) * (years_elapsed / total_years)
    
    target_cum_units = int(tam_yr * (target_penetration_yr / 100.0))
    
    # Annual sales = target cumulative - (baseline + historical + previous forecast years)
    cum_forecast_so_far = sum(forecast_units.get(y, 0) for y in range(2026, yr))
    forecast_units[yr] = max(0, target_cum_units - cum_units_baseline - cum_units_historical - cum_forecast_so_far)

# Engineering Hours Forecast (2026-2035) - growth-based (keep as-is, not TAM-based)
eng_hours_forecast = {}
for i, yr in enumerate(range(2026, 2036)):
    eng_hours_forecast[yr] = max(0, eng_hours_start + (i * eng_hours_growth))

# 510 Mustang winglet forecast (2028-2035) - Annual sales ramping to achieve cumulative penetration target
forecast_units_510 = {}
for i, yr in enumerate(range(2028, 2036)):
    years_elapsed = yr - 2028
    total_years = 7  # 2028 to 2035
    
    # Penetration ramps from 0% to target by 2035
    target_penetration_yr = (penetration_510_2035 / 100.0) * (years_elapsed / total_years)
    target_cum_units = int(base_tam_510 * target_penetration_yr)
    
    # Annual sales = target cumulative - previous forecast years
    cum_forecast_so_far = sum(forecast_units_510.get(y, 0) for y in range(2028, yr))
    forecast_units_510[yr] = max(0, target_cum_units - cum_forecast_so_far)

# Starlink STC forecast (2026-2035) - Annual sales ramping to achieve cumulative penetration target
starlink_forecast = {}
for i, yr in enumerate(range(2026, 2036)):
    years_elapsed = yr - 2026
    total_years = 9  # 2026 to 2035
    
    # Calculate target cumulative units for this year
    tam_yr = base_tam_starlink + ((new_525_per_year + new_cj4_per_year) * (yr - 2016))
    
    # Penetration ramps from 0% to target by 2035
    target_penetration_yr = (penetration_starlink_2035 / 100.0) * (years_elapsed / total_years)
    target_cum_units = int(tam_yr * target_penetration_yr)
    
    # Annual sales = target cumulative - previous forecast years
    cum_forecast_so_far = sum(starlink_forecast.get(y, 0) for y in range(2026, yr))
    starlink_forecast[yr] = max(0, target_cum_units - cum_forecast_so_far)

# Challenger 604 family winglet forecast (2027-2035) - Annual sales ramping to achieve cumulative penetration target
forecast_units_604 = {}
for i, yr in enumerate(range(2027, 2036)):
    years_elapsed = yr - 2027
    total_years = 8  # 2027 to 2035
    
    # Calculate target cumulative units for this year
    # TAM grows from 1200 (as of 2025) + new production from 2026 onwards
    years_since_2025 = max(0, yr - 2025)
    tam_yr = base_tam_604 + (new_604_per_year * years_since_2025)
    
    # Penetration ramps from 0% to target by 2035
    target_penetration_yr = (penetration_604_2035 / 100.0) * (years_elapsed / total_years)
    target_cum_units = int(tam_yr * target_penetration_yr)
    
    # Annual sales = target cumulative - previous forecast years
    cum_forecast_so_far = sum(forecast_units_604.get(y, 0) for y in range(2027, yr))
    forecast_units_604[yr] = max(0, target_cum_units - cum_forecast_so_far)

# 510 Investor funding (all in 2026)
# $5M for project/certification + $1M for inventory = $6M usable for operations
# $2M for aircraft asset (not included, not usable for operations)
investor_510_funding = 6.0  # $6M total usable for operations in 2026

# Operational costs based on 2025 actuals (excludes depreciation, amortization, and interest)
# Annual costs in $M
opex_ga_annual = 1.750  # G&A
opex_marketing_annual = 0.815  # Marketing
opex_engineering_annual = 1.800  # Engineering
opex_production_annual = 1.120  # Production/Install

# Total OpEx per year
opex_2025_total = opex_ga_annual + opex_marketing_annual + opex_engineering_annual + opex_production_annual

# OpEx schedule ($M/year)
# Historical years use legacy estimates, 2025 baseline, then grow at opex_growth_rate
opex = {
    2016: 2, 2017: 3, 2018: 5, 2019: 4, 2020: 4, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 
    2025: round(opex_2025_total, 2),
}
# Apply growth rate from 2026 onwards
for i, yr in enumerate(range(2026, 2036)):
    opex[yr] = round(opex_2025_total * ((1 + opex_growth_rate) ** (i + 1)), 2)

# Years
years = list(range(2016, 2036))  # 20 years

# Build assumptions table
assumptions_rows = [
    {'Assumption': 'Base 525 Winglet Price (2026)', 'Value': f"{kit_price_base}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Starting 525 winglet price in 2026'},
    {'Assumption': 'Annual 525 Price Escalation', 'Value': f"{kit_price_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to 525 winglet price each year'},
    {'Assumption': 'Base COGS per 525 Winglet (2026)', 'Value': f"{base_cogs_k}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Starting COGS per 525 winglet in 2026'},
    {'Assumption': 'Annual 525 COGS Escalation', 'Value': f"{cogs_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to 525 COGS per winglet each year'},
    {'Assumption': '525 Winglet TAM', 'Value': '2,200 + 60/yr', 'Units': 'aircraft', 'Type': 'Hardwired', 'Notes': '2,200 existing 525 series (M2/CJ1/CJ2/CJ3/CJ3+) + 60 new per year (excludes CJ4/525C)'},
    {'Assumption': 'Base 510 Winglet Price (2028)', 'Value': f"{kit_510_price_base}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Fixed 510 winglet price starting 2028'},
    {'Assumption': '510 Winglet COGS', 'Value': f"{kit_510_cogs}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Fixed COGS per 510 winglet'},
    {'Assumption': '510 Winglet TAM', 'Value': '450', 'Units': 'aircraft', 'Type': 'Hardwired', 'Notes': 'Total 510 Mustang aircraft (no longer produced)'},
    {'Assumption': '510 Investor Funding', 'Value': '6.0', 'Units': '$M', 'Type': 'Hardwired', 'Notes': 'Investment in 2026 for 510 project ($5M) and inventory ($1M), flows to bottom line'},
    {'Assumption': 'Base 604 Winglet Price (2027)', 'Value': f"{kit_604_price_base}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Starting 604 winglet price in 2027'},
    {'Assumption': 'Annual 604 Price Escalation', 'Value': f"{kit_604_price_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to 604 winglet price each year'},
    {'Assumption': '604 Winglet COGS', 'Value': f"{kit_604_cogs}", 'Units': '$k/winglet', 'Type': 'Slider', 'Notes': 'Fixed COGS per 604 winglet'},
    {'Assumption': '604 Winglet TAM', 'Value': '1,200 + 35/yr', 'Units': 'aircraft', 'Type': 'Hardwired', 'Notes': '1,200 existing Challenger 604/605/650 as of 2025 + 35 new Challenger 650 per year from 2026'},
    {'Assumption': '604 Certification Cost', 'Value': f"{kit_604_cert_cost:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': '2-year certification project (2026-2027), $8M per year'},
    {'Assumption': 'Engineering Billing Rate', 'Value': f"{eng_rate}", 'Units': '$/hr', 'Type': 'Slider', 'Notes': 'Revenue per billable hour'},
    {'Assumption': 'Engineering Cost per Hour', 'Value': f"{eng_cost_per_hour}", 'Units': '$/hr', 'Type': 'Slider', 'Notes': 'Direct cost per billable hour'},
    {'Assumption': 'Engineering Overhead', 'Value': f"{eng_overhead_pct * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Overhead applied to engineering direct costs'},
    {'Assumption': 'Base Starlink Price (2026)', 'Value': f"{starlink_price_base}", 'Units': '$k/unit', 'Type': 'Slider', 'Notes': 'Starting Starlink STC price in 2026'},
    {'Assumption': 'Annual Starlink Price Escalation', 'Value': f"{starlink_price_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to Starlink price each year'},
    {'Assumption': 'Starlink COGS per Unit', 'Value': f"{starlink_cogs}", 'Units': '$k/unit', 'Type': 'Slider', 'Notes': 'Hardware cost per Starlink unit'},
    {'Assumption': 'Starlink Install Cost per Unit', 'Value': f"{starlink_install}", 'Units': '$k/unit', 'Type': 'Slider', 'Notes': 'Installation cost per Starlink unit'},
    {'Assumption': 'Starlink Engineering Commitment', 'Value': f"{starlink_eng_commitment}", 'Units': '$k', 'Type': 'Slider', 'Notes': 'One-time engineering cost in 2026 for STC certification'},
    {'Assumption': 'Starlink Addressable Market', 'Value': '2,650 + 91/yr', 'Units': 'aircraft', 'Type': 'Hardwired', 'Notes': 'All 525 series + 510 aircraft + 60 new 525 + 31 new CJ4 (525C) per year'},
    {'Assumption': 'Max Debt Available', 'Value': f"{debt_amount:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': 'Debt facility cap'},
    {'Assumption': 'Debt APR', 'Value': f"{debt_apr * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to outstanding debt balance'},
    {'Assumption': 'Debt Term', 'Value': f"{debt_term_years}", 'Units': 'Years', 'Type': 'Slider', 'Notes': 'Debt amortization period'},
    {'Assumption': 'Income Tax Rate', 'Value': f"{tax_rate * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Taxes apply only when taxable income is positive'},
    {'Assumption': 'WACC', 'Value': f"{wacc * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used to discount unlevered free cash flows in DCF'},
    {'Assumption': 'Terminal Growth Rate', 'Value': f"{terminal_growth * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used for terminal value if WACC > terminal growth'},
    {'Assumption': 'Model Years', 'Value': f"{years[0]}-{years[-1]}", 'Units': 'Years', 'Type': 'Hardwired', 'Notes': 'Annual model projection period (10 historical + 10 forecast)'},
    {'Assumption': 'OpEx - G&A (2025)', 'Value': f"{opex_ga_annual:.3f}", 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'General & Administrative costs (excludes D&A, interest)'},
    {'Assumption': 'OpEx - Marketing (2025)', 'Value': f"{opex_marketing_annual:.3f}", 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'Marketing costs including sales team'},
    {'Assumption': 'OpEx - Engineering (2025)', 'Value': f"{opex_engineering_annual:.3f}", 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'Engineering department costs'},
    {'Assumption': 'OpEx - Production/Install (2025)', 'Value': f"{opex_production_annual:.3f}", 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'Production and installation costs (excludes inventory build)'},
    {'Assumption': 'Total OpEx (2025+)', 'Value': f"{opex_2025_total:.3f}", 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'Sum of G&A, Marketing, Engineering, and Production costs'},
    {'Assumption': 'Annual OpEx Growth Rate', 'Value': f"{opex_growth_rate * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to OpEx each year starting from 2026'},
]

assumptions_df = pd.DataFrame(assumptions_rows, columns=['Assumption', 'Value', 'Units', 'Type', 'Notes'])

# Calculations
annual_data = {}

cum_cash = 0.0
debt_balance = 0.0
debt_drawn_total = 0.0
quarterly_debt_payment = None

debt_rate_annual = float(debt_apr)
term_years = int(debt_term_years)

# Market tracking
cum_winglet_units = 218  # 218 of 2200 already installed before 2016
cum_winglet_510_units = 0
cum_winglet_604_units = 0
cum_starlink_units = 0

for yr in years:
    # Winglet sales
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
    
    # Cumulative 525 winglet tracking
    cum_winglet_units += units
    winglet_525_tam = base_tam_525 + (new_525_per_year * max(0, yr - 2016))
    winglet_525_penetration = (cum_winglet_units / winglet_525_tam * 100) if winglet_525_tam > 0 else 0.0
    
    # 510 Mustang winglet sales (starts 2028)
    if yr in forecast_units_510:
        units_510 = forecast_units_510[yr]
    else:
        units_510 = 0
    
    # 510 pricing (fixed, no escalation)
    kit_510_price = kit_510_price_base * 1000.0
    kit_510_cogs_unit = kit_510_cogs * 1000.0
    
    kit_510_revenue = units_510 * kit_510_price / 1e6
    kit_510_cogs_total = units_510 * kit_510_cogs_unit / 1e6
    
    # Cumulative 510 winglet tracking
    if yr >= 2028:
        cum_winglet_510_units += units_510
    winglet_510_penetration = (cum_winglet_510_units / base_tam_510 * 100) if base_tam_510 > 0 else 0.0
    
    # 510 investor funding (all in 2026)
    investor_funding_510 = 0.0
    if yr == 2026:
        investor_funding_510 = investor_510_funding  # $6M for operations
    
    # Engineering services
    if yr in eng_hours_forecast:
        eng_hours = eng_hours_forecast[yr] * 1000.0  # Convert from k to actual hours
    else:
        eng_hours = 0.0
    
    eng_revenue = eng_hours * eng_rate / 1e6
    eng_direct_cost = eng_hours * eng_cost_per_hour / 1e6
    eng_overhead = eng_direct_cost * eng_overhead_pct
    eng_cogs_total = eng_direct_cost + eng_overhead
    
    # Challenger 604 family winglet sales (starts 2027)
    if yr in forecast_units_604:
        units_604 = forecast_units_604[yr]
    else:
        units_604 = 0
    
    # 604 pricing (starts in 2027, escalates annually)
    kit_604_year_idx = max(0, yr - 2027)
    kit_604_price = (kit_604_price_base * 1000.0) * ((1 + kit_604_price_escalation) ** kit_604_year_idx)
    kit_604_cogs_unit = kit_604_cogs * 1000.0
    
    kit_604_revenue = units_604 * kit_604_price / 1e6
    kit_604_cogs_total = units_604 * kit_604_cogs_unit / 1e6
    
    # Cumulative 604 winglet tracking
    if yr >= 2027:
        cum_winglet_604_units += units_604
    # TAM calculation: 1200 as of 2025, then add new production from 2026 onwards
    years_since_2025 = max(0, yr - 2025)
    winglet_604_tam = base_tam_604 + (new_604_per_year * years_since_2025)
    winglet_604_penetration = (cum_winglet_604_units / winglet_604_tam * 100) if winglet_604_tam > 0 else 0.0
    
    # 604 Certification costs (2-year project: 2026-2027, $8M per year)
    cert_604_cost = 0.0
    if yr == 2026:
        cert_604_cost = kit_604_cert_cost / 2.0  # Half in 2026
    elif yr == 2027:
        cert_604_cost = kit_604_cert_cost / 2.0  # Half in 2027
    
    # Starlink STC sales
    if yr in starlink_forecast:
        starlink_units = starlink_forecast[yr]
    else:
        starlink_units = 0
    
    # Starlink pricing starts in 2026
    starlink_year_idx = max(0, yr - 2026)
    starlink_price = (starlink_price_base * 1000.0) * ((1 + starlink_price_escalation) ** starlink_year_idx)
    starlink_cogs_per_unit = (starlink_cogs + starlink_install) * 1000.0
    
    starlink_revenue = starlink_units * starlink_price / 1e6
    starlink_cogs_total = starlink_units * starlink_cogs_per_unit / 1e6
    
    # Starlink engineering commitment (one-time in 2026)
    starlink_eng_cost = (starlink_eng_commitment / 1000.0) if yr == 2026 else 0.0
    
    # Cumulative Starlink tracking (only from 2026, includes all 525 series + 510)
    if yr >= 2026:
        cum_starlink_units += starlink_units
    # Starlink TAM: base 525+510 aircraft + new 525 series (90/yr) + new CJ4 (31/yr) = 121/yr total
    starlink_tam = base_tam_starlink + ((new_525_per_year + new_cj4_per_year) * max(0, yr - 2016))
    starlink_penetration = (cum_starlink_units / starlink_tam * 100) if starlink_tam > 0 else 0.0
    
    # Apply project enable/disable flags
    if not enable_525:
        kit_revenue = 0.0
        kit_cogs_total = 0.0
    if not enable_510:
        kit_510_revenue = 0.0
        kit_510_cogs_total = 0.0
    if not enable_604:
        kit_604_revenue = 0.0
        kit_604_cogs_total = 0.0
    if not enable_engineering:
        eng_revenue = 0.0
        eng_cogs_total = 0.0
    if not enable_starlink:
        starlink_revenue = 0.0
        starlink_cogs_total = 0.0
        starlink_eng_cost = 0.0
    
    # Total revenue and COGS
    revenue = kit_revenue + kit_510_revenue + kit_604_revenue + eng_revenue + starlink_revenue
    cogs = kit_cogs_total + kit_510_cogs_total + kit_604_cogs_total + eng_cogs_total + starlink_cogs_total + starlink_eng_cost + cert_604_cost
    gross_profit = revenue - cogs
    
    # OpEx
    opex_yr = opex.get(yr, 8.0)
    
    # EBITDA
    ebitda = gross_profit - opex_yr
    
    # Debt service (simple annual payment if debt exists)
    if yr == 2026 and debt_amount > 0:
        # Draw debt in 2026 if needed
        if ebitda < 0:
            debt_draw = min(debt_amount, abs(ebitda))
            debt_balance = debt_draw
            debt_drawn_total = debt_draw
        else:
            debt_draw = 0.0
    else:
        debt_draw = 0.0
    
    # Debt interest and principal
    if debt_balance > 0:
        debt_interest = debt_balance * debt_rate_annual
        if quarterly_debt_payment is None and term_years > 0:
            # Calculate annual payment (amortizing loan)
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
    
    # Taxes
    taxable_income = ebitda - debt_interest
    taxes = max(0.0, taxable_income) * tax_rate
    
    # Free cash flow
    fcf_after_tax = ebitda - taxes
    net_cash_after_debt = fcf_after_tax + debt_draw - debt_payment + investor_funding_510
    
    # Reset cumulative cash at start of 2026
    if yr == 2026:
        cum_cash = 0.0
    
    cum_cash += net_cash_after_debt
    
    annual_data[yr] = {
        '525 Winglet Units': int(units),
        'Cum 525 Winglets': int(cum_winglet_units),
        '525 Mkt Pen (%)': round(winglet_525_penetration, 1),
        '525 Revenue ($M)': round(kit_revenue, 2),
        '525 COGS ($M)': round(kit_cogs_total, 2),
        '510 Winglet Units': int(units_510),
        'Cum 510 Winglets': int(cum_winglet_510_units),
        '510 Mkt Pen (%)': round(winglet_510_penetration, 1),
        '510 Revenue ($M)': round(kit_510_revenue, 2),
        '510 COGS ($M)': round(kit_510_cogs_total, 2),
        '604 Winglet Units': int(units_604),
        'Cum 604 Winglets': int(cum_winglet_604_units),
        '604 Mkt Pen (%)': round(winglet_604_penetration, 1),
        '604 Revenue ($M)': round(kit_604_revenue, 2),
        '604 COGS ($M)': round(kit_604_cogs_total, 2),
        '604 Cert Cost ($M)': round(cert_604_cost, 2),
        'Eng Hours (k)': round(eng_hours / 1000.0, 1) if yr >= 2026 else 0.0,
        'Eng Revenue ($M)': round(eng_revenue, 2),
        'Eng COGS ($M)': round(eng_cogs_total, 2),
        'Starlink Units': int(starlink_units),
        'Cum Starlink Units': int(cum_starlink_units),
        'Starlink Mkt Pen (%)': round(starlink_penetration, 1),
        'Starlink Revenue ($M)': round(starlink_revenue, 2),
        'Starlink COGS ($M)': round(starlink_cogs_total, 2),
        'Starlink Eng Cost ($M)': round(starlink_eng_cost, 2),
        'Total Revenue ($M)': round(revenue, 2),
        'Total COGS ($M)': round(cogs, 2),
        'Gross Profit ($M)': round(gross_profit, 2),
        'OpEx ($M)': round(opex_yr, 1),
        'EBITDA ($M)': round(ebitda, 2),
        'Debt Interest ($M)': round(debt_interest, 2),
        'Taxes ($M)': round(taxes, 2),
        'FCF After Tax ($M)': round(fcf_after_tax, 2),
        'Debt Draw ($M)': round(debt_draw, 2),
        'Debt Payment ($M)': round(debt_payment, 2),
        'Debt Balance ($M)': round(debt_balance, 2),
        'Cumulative Cash ($M)': round(cum_cash, 2),
    }

df = pd.DataFrame(annual_data).T

# P&L
net_income = df['EBITDA ($M)'] - df['Debt Interest ($M)'] - df['Taxes ($M)']
pl_df = df[['Total Revenue ($M)', 'Total COGS ($M)', 'Gross Profit ($M)', 'OpEx ($M)', 'EBITDA ($M)', 'Debt Interest ($M)', 'Taxes ($M)']].copy()
pl_df['Net Income ($M)'] = net_income.round(2)

# Balance Sheet
equity_paid_in = 0.0  # No equity in this model
retained_earnings = net_income.cumsum()
bs_df = pd.DataFrame({
    'Cash ($M)': df['Cumulative Cash ($M)'],
    'Debt Balance ($M)': df['Debt Balance ($M)'],
    'Retained Earnings ($M)': retained_earnings,
}, index=df.index)
bs_df['Total Assets ($M)'] = bs_df['Cash ($M)']
bs_df['Total Liab + Equity ($M)'] = bs_df['Debt Balance ($M)'] + bs_df['Retained Earnings ($M)']

# Cash Flow Statement
operating_cf = df['EBITDA ($M)'] - df['Taxes ($M)']
investing_cf = 0.0  # No CapEx in this simplified model
financing_cf = df['Debt Draw ($M)'] - df['Debt Payment ($M)']
cash_net_change = operating_cf + financing_cf
cf_df = pd.DataFrame({
    'Operating CF ($M)': operating_cf,
    'Financing CF ($M)': financing_cf,
    'Net Change in Cash ($M)': cash_net_change,
    'Ending Cash ($M)': df['Cumulative Cash ($M)'],
}, index=df.index)

# DCF Analysis
unlevered_taxes = (df['EBITDA ($M)'].clip(lower=0.0) * tax_rate).astype(float)
unlevered_fcf = (df['EBITDA ($M)'] - unlevered_taxes).astype(float)

discount_year0 = int(df.index.min())
discount_t = (df.index - discount_year0 + 1).astype(int)
discount_factor = pd.Series((1 / (1 + wacc) ** discount_t).astype(float), index=df.index)
pv_fcf = (unlevered_fcf * discount_factor).astype(float).round(2)

tv = np.nan
pv_tv = np.nan
if wacc > terminal_growth:
    tv = float(unlevered_fcf.iloc[-1]) * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv * float(discount_factor.iloc[-1])

dcf_df = pd.DataFrame({
    'Unlevered FCF ($M)': unlevered_fcf.round(2),
    'Discount Factor': discount_factor.round(4),
    'PV of FCF ($M)': pv_fcf,
}, index=df.index)

pv_explicit = float(pv_fcf.sum())
enterprise_value = pv_explicit + (pv_tv if not np.isnan(pv_tv) else 0.0)

ev_placeholder.metric(label="Enterprise Value ($M)", value=f"{enterprise_value:,.1f}")

dcf_summary_df = pd.DataFrame({
    'PV Explicit FCF ($M)': [round(pv_explicit, 1)],
    'Terminal Value ($M)': [round(0.0 if np.isnan(tv) else tv, 1)],
    'PV Terminal Value ($M)': [round(0.0 if np.isnan(pv_tv) else pv_tv, 1)],
    'Enterprise Value ($M)': [round(enterprise_value, 1)],
})

# Filter to show only 2026 onwards for output
forecast_years_output = list(range(2026, 2036))
df_output = df.loc[forecast_years_output]
pl_df_output = pl_df.loc[forecast_years_output]
bs_df_output = bs_df.loc[forecast_years_output]
cf_df_output = cf_df.loc[forecast_years_output]
dcf_df_output = dcf_df.loc[forecast_years_output]

# Display main table
st.header('Annual Financial Performance & Market Penetration (2026-2035)')
st.dataframe(df_output, use_container_width=True)

st.header('Three-Statement Output (2026-2035)')
st.subheader('P&L')
st.dataframe(pl_df_output, use_container_width=True)
st.subheader('Balance Sheet')
st.dataframe(bs_df_output, use_container_width=True)
st.subheader('Statement of Cash Flows')
st.dataframe(cf_df_output, use_container_width=True)

st.header('DCF Analysis (2026-2035)')
st.dataframe(dcf_df_output, use_container_width=True)

st.subheader('DCF Supporting Information')
st.write(f"Discount base year: {discount_year0}")
st.write(f"WACC: {wacc * 100:.2f}%")
st.write(f"Terminal growth rate: {terminal_growth * 100:.2f}%")
st.write(f"PV of explicit period FCF ($M): {pv_explicit:.1f}")
st.write(f"PV of terminal value ($M): {0.0 if np.isnan(pv_tv) else pv_tv:.1f}")
st.dataframe(dcf_summary_df, use_container_width=True)

st.header('Financial Projection Plots')

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
x = np.arange(len(years))
width = 0.25

ax.bar(x - width, df['Total Revenue ($M)'], width=width, color='orange', label='Revenue')
ax.bar(x, df['Gross Profit ($M)'], width=width, color='green', label='Gross Profit')
ax.bar(x + width, df['FCF After Tax ($M)'], width=width, color='blue', label='FCF After Tax')
ax.set_title('Annual Revenue, Gross Profit, and Free Cash Flow')
ax.set_ylabel('$M')
ax.set_xlabel('Year')
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, ha='right')
ax.legend(ncol=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.axvline(x=9.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Historical/Forecast Split')

plt.tight_layout()
st.pyplot(fig)

# Revenue breakdown chart (2026 onwards)
forecast_years = list(range(2026, 2036))
df_forecast = df.loc[forecast_years]
fig_rev, ax_rev = plt.subplots(1, 1, figsize=(14, 6))
x_forecast = np.arange(len(forecast_years))
width_rev = 0.2

ax_rev.bar(x_forecast - 2*width_rev, df_forecast['525 Revenue ($M)'], width=width_rev, color='#FF6B6B', label='525 Winglet Revenue')
ax_rev.bar(x_forecast - width_rev, df_forecast['510 Revenue ($M)'], width=width_rev, color='#FF9999', label='510 Winglet Revenue')
ax_rev.bar(x_forecast, df_forecast['604 Revenue ($M)'], width=width_rev, color='#FFA07A', label='604 Winglet Revenue')
ax_rev.bar(x_forecast + width_rev, df_forecast['Eng Revenue ($M)'], width=width_rev, color='#4ECDC4', label='Engineering Revenue')
ax_rev.bar(x_forecast + 2*width_rev, df_forecast['Starlink Revenue ($M)'], width=width_rev, color='#95E1D3', label='Starlink Revenue')
ax_rev.set_title('Revenue Breakdown by Source (2026-2035)')
ax_rev.set_ylabel('$M')
ax_rev.set_xlabel('Year')
ax_rev.set_xticks(x_forecast)
ax_rev.set_xticklabels(forecast_years, rotation=45, ha='right')
ax_rev.legend(ncol=5, fontsize=9)
ax_rev.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig_rev)

cumulative_cash = df['FCF After Tax ($M)'].copy()
for i, yr in enumerate(years):
    if yr < 2026:
        cumulative_cash.iloc[i] = 0.0
    elif yr == 2026:
        cumulative_cash.iloc[i] = df['FCF After Tax ($M)'].iloc[i]
    else:
        cumulative_cash.iloc[i] = cumulative_cash.iloc[i-1] + df['FCF After Tax ($M)'].iloc[i]

# Filter to show only 2026 onwards
forecast_years_cum = list(range(2026, 2036))
cumulative_cash_forecast = cumulative_cash.loc[forecast_years_cum]

fig_cum, ax_cum = plt.subplots(1, 1, figsize=(14, 4))
ax_cum.plot(forecast_years_cum, cumulative_cash_forecast, color='purple', marker='o', linewidth=2, label='Cumulative FCF After Tax (from 2026)')
ax_cum.axhline(0, color='black', linewidth=1, alpha=0.4)
ax_cum.set_title('Cumulative Cash (Cumulative FCF After Tax from 2026)')
ax_cum.set_ylabel('$M')
ax_cum.set_xlabel('Year')
ax_cum.set_xticks(forecast_years_cum)
ax_cum.legend()
ax_cum.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig_cum)

st.header('Assumptions Appendix')
st.dataframe(assumptions_df, use_container_width=True)
