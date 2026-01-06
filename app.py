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
_img_a320 = _asset_dir / 'A320Tam.jpg'
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
with col_main:
    pass

if mode == 'Sensitivity Study (3 Drivers, 1 Output)':
    render_sensitivity_app(baseline_params=None, show_title=False)
    st.stop()

# Sidebar inputs
st.sidebar.header('Kit Sales Revenue')
kit_price_base = st.sidebar.slider('Base Kit Price (2016, $k)', min_value=100, max_value=2000, value=800, step=50)
kit_price_escalation = st.sidebar.slider('Annual Price Escalation (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100
base_cogs_k = st.sidebar.slider('Base COGS per Kit (2016, $k)', min_value=50, max_value=1000, value=400, step=25)
cogs_escalation = st.sidebar.slider('Annual COGS Escalation (%)', min_value=0.0, max_value=10.0, value=3.0, step=0.5) / 100

st.sidebar.header('Kit Sales Forecast (2026-2035)')
units_2026 = st.sidebar.slider('2026 Units', min_value=0, max_value=100, value=20, step=5)
units_2027 = st.sidebar.slider('2027 Units', min_value=0, max_value=100, value=25, step=5)
units_2028 = st.sidebar.slider('2028 Units', min_value=0, max_value=100, value=30, step=5)
units_2029 = st.sidebar.slider('2029 Units', min_value=0, max_value=100, value=35, step=5)
units_2030 = st.sidebar.slider('2030 Units', min_value=0, max_value=100, value=40, step=5)
units_2031_2035 = st.sidebar.slider('2031-2035 Units (annual)', min_value=0, max_value=100, value=40, step=5)

st.sidebar.header('Engineering Services Revenue')
eng_rate = st.sidebar.slider('Billing Rate ($/hr)', min_value=50, max_value=500, value=200, step=10)
eng_cost_per_hour = st.sidebar.slider('Cost per Billable Hour ($/hr)', min_value=20, max_value=300, value=120, step=10)
eng_overhead_pct = st.sidebar.slider('Engineering Overhead (%)', min_value=0.0, max_value=100.0, value=20.0, step=5.0) / 100

st.sidebar.header('Engineering Hours Forecast (2026-2035)')
eng_hours_2026 = st.sidebar.slider('2026 Hours (k)', min_value=0, max_value=50, value=5, step=1)
eng_hours_2027 = st.sidebar.slider('2027 Hours (k)', min_value=0, max_value=50, value=8, step=1)
eng_hours_2028 = st.sidebar.slider('2028 Hours (k)', min_value=0, max_value=50, value=10, step=1)
eng_hours_2029 = st.sidebar.slider('2029 Hours (k)', min_value=0, max_value=50, value=12, step=1)
eng_hours_2030 = st.sidebar.slider('2030 Hours (k)', min_value=0, max_value=50, value=15, step=1)
eng_hours_2031_2035 = st.sidebar.slider('2031-2035 Hours (k, annual)', min_value=0, max_value=50, value=15, step=1)

st.sidebar.header('Financial')
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

# Forecast units (2026-2035)
forecast_units = {
    2026: units_2026,
    2027: units_2027,
    2028: units_2028,
    2029: units_2029,
    2030: units_2030,
    2031: units_2031_2035,
    2032: units_2031_2035,
    2033: units_2031_2035,
    2034: units_2031_2035,
    2035: units_2031_2035,
}

# Engineering hours forecast (2026-2035, in thousands)
eng_hours_forecast = {
    2026: eng_hours_2026,
    2027: eng_hours_2027,
    2028: eng_hours_2028,
    2029: eng_hours_2029,
    2030: eng_hours_2030,
    2031: eng_hours_2031_2035,
    2032: eng_hours_2031_2035,
    2033: eng_hours_2031_2035,
    2034: eng_hours_2031_2035,
    2035: eng_hours_2031_2035,
}

# OpEx schedule ($M/year)
opex = {
    2016: 2, 2017: 3, 2018: 5, 2019: 4, 2020: 4, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 4,
    2026: 5, 2027: 6, 2028: 7, 2029: 8, 2030: 8, 2031: 8, 2032: 8, 2033: 8, 2034: 8, 2035: 8
}

# Years
years = list(range(2016, 2036))  # 20 years

# Build assumptions table
assumptions_rows = [
    {'Assumption': 'Base Kit Price (2016)', 'Value': f"{kit_price_base}", 'Units': '$k/kit', 'Type': 'Slider', 'Notes': 'Starting kit price in 2016'},
    {'Assumption': 'Annual Price Escalation', 'Value': f"{kit_price_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to kit price each year'},
    {'Assumption': 'Base COGS per Kit (2016)', 'Value': f"{base_cogs_k}", 'Units': '$k/kit', 'Type': 'Slider', 'Notes': 'Starting COGS per kit in 2016'},
    {'Assumption': 'Annual COGS Escalation', 'Value': f"{cogs_escalation * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to COGS per kit each year'},
    {'Assumption': 'Engineering Billing Rate', 'Value': f"{eng_rate}", 'Units': '$/hr', 'Type': 'Slider', 'Notes': 'Revenue per billable hour'},
    {'Assumption': 'Engineering Cost per Hour', 'Value': f"{eng_cost_per_hour}", 'Units': '$/hr', 'Type': 'Slider', 'Notes': 'Direct cost per billable hour'},
    {'Assumption': 'Engineering Overhead', 'Value': f"{eng_overhead_pct * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Overhead applied to engineering direct costs'},
    {'Assumption': 'Max Debt Available', 'Value': f"{debt_amount:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': 'Debt facility cap'},
    {'Assumption': 'Debt APR', 'Value': f"{debt_apr * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to outstanding debt balance'},
    {'Assumption': 'Debt Term', 'Value': f"{debt_term_years}", 'Units': 'Years', 'Type': 'Slider', 'Notes': 'Debt amortization period'},
    {'Assumption': 'Income Tax Rate', 'Value': f"{tax_rate * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Taxes apply only when taxable income is positive'},
    {'Assumption': 'WACC', 'Value': f"{wacc * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used to discount unlevered free cash flows in DCF'},
    {'Assumption': 'Terminal Growth Rate', 'Value': f"{terminal_growth * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used for terminal value if WACC > terminal growth'},
    {'Assumption': 'Model Years', 'Value': f"{years[0]}-{years[-1]}", 'Units': 'Years', 'Type': 'Hardwired', 'Notes': 'Annual model projection period (10 historical + 10 forecast)'},
    {'Assumption': 'OpEx Schedule', 'Value': 'See model', 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'Operating expenses by year'},
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
        eng_hours = eng_hours_forecast[yr] * 1000.0  # Convert from k to actual hours
    else:
        eng_hours = 0.0
    
    eng_revenue = eng_hours * eng_rate / 1e6
    eng_direct_cost = eng_hours * eng_cost_per_hour / 1e6
    eng_overhead = eng_direct_cost * eng_overhead_pct
    eng_cogs_total = eng_direct_cost + eng_overhead
    
    # Total revenue and COGS
    revenue = kit_revenue + eng_revenue
    cogs = kit_cogs_total + eng_cogs_total
    gross_profit = revenue - cogs
    
    # OpEx
    opex_yr = opex.get(yr, 8.0)
    
    # EBITDA
    ebitda = gross_profit - opex_yr
    
    # Debt service (simple annual payment if debt exists)
    if yr == 2016 and debt_amount > 0:
        # Draw debt in first year if needed
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
    net_cash_after_debt = fcf_after_tax + debt_draw - debt_payment
    cum_cash += net_cash_after_debt
    
    annual_data[yr] = {
        'Kit Units': int(units),
        'Kit Revenue ($M)': round(kit_revenue, 2),
        'Kit COGS ($M)': round(kit_cogs_total, 2),
        'Eng Hours (k)': round(eng_hours / 1000.0, 1) if yr >= 2026 else 0.0,
        'Eng Revenue ($M)': round(eng_revenue, 2),
        'Eng COGS ($M)': round(eng_cogs_total, 2),
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

# Display main table
st.dataframe(df, use_container_width=True)

st.header('Three-Statement Output')
st.subheader('P&L')
st.dataframe(pl_df, use_container_width=True)
st.subheader('Balance Sheet')
st.dataframe(bs_df, use_container_width=True)
st.subheader('Statement of Cash Flows')
st.dataframe(cf_df, use_container_width=True)

st.header('DCF Analysis')
st.dataframe(dcf_df, use_container_width=True)

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

cumulative_cash = df['FCF After Tax ($M)'].cumsum()
fig_cum, ax_cum = plt.subplots(1, 1, figsize=(14, 4))
ax_cum.plot(years, cumulative_cash, color='purple', marker='o', linewidth=2, label='Cumulative FCF After Tax')
ax_cum.axhline(0, color='black', linewidth=1, alpha=0.4)
ax_cum.axvline(x=2025.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Historical/Forecast Split')
ax_cum.set_title('Cumulative Cash (Cumulative FCF After Tax)')
ax_cum.set_ylabel('$M')
ax_cum.set_xlabel('Year')
ax_cum.legend()
ax_cum.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig_cum)

st.header('Assumptions Appendix')
st.dataframe(assumptions_df, use_container_width=True)
