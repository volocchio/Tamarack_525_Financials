# Tamarack 525 Financial Model

A comprehensive financial projection model for Tamarack's Citation 525 (CJ) series aircraft modification program, covering 2016-2035 (10 years historical + 10 years forecast).

## Overview

This Streamlit-based financial model projects revenue, costs, cash flows, and enterprise value for two revenue streams:
1. **Kit Sales** - Fixed-price aircraft modification kits with historical sales data (2016-2025) and configurable forecasts (2026-2035)
2. **Engineering Services** - Billable hours-based consulting/engineering revenue with cost-per-hour and overhead modeling

## Features

### Two Operating Modes

#### 1. Standalone Model
- Full 20-year financial projection (2016-2035)
- Integrated P&L, Balance Sheet, and Cash Flow statements
- DCF-based enterprise valuation
- Interactive charts showing revenue, profitability, and cumulative cash
- Configurable assumptions via sidebar sliders

#### 2. Sensitivity Study (3-Driver Goal Setting)
- Select any 3 input drivers (e.g., units sold, kit price, engineering hours)
- Choose 1 output metric (e.g., Enterprise Value, EBITDA, Ending Cash)
- Generate heatmap slices showing how combinations of inputs affect the target metric
- Ideal for setting sales targets and pricing strategies to hit financial goals

### Revenue Streams

**Kit Sales**
- Historical units: 2016 (4), 2017 (24), 2018 (52), 2019 (18), 2020 (22), 2021 (26), 2022 (24), 2023 (16), 2024 (14), 2025 (18)
- Base price (2016): Configurable, default $800k/kit
- Annual price escalation: Configurable, default 2%
- COGS escalation: Configurable, default 3%
- Forecast units (2026-2035): Fully configurable via sliders

**Engineering Services**
- Billing rate: Configurable ($/hr), default $200/hr
- Cost per billable hour: Configurable ($/hr), default $120/hr
- Overhead percentage: Applied to direct costs, default 20%
- Forecast hours (2026-2035): Fully configurable in thousands of hours

### Financial Outputs

**Three-Statement Model**
- **P&L**: Revenue, COGS, Gross Profit, OpEx, EBITDA, Interest, Taxes, Net Income
- **Balance Sheet**: Cash, Debt, Retained Earnings
- **Cash Flow**: Operating, Financing, Net Change, Ending Cash

**DCF Valuation**
- Unlevered free cash flow projection
- Configurable WACC and terminal growth rate
- Terminal value calculation (if WACC > terminal growth)
- Present value of explicit period + terminal value = Enterprise Value

**Debt Financing**
- Optional debt facility (default $20M)
- Amortizing loan structure with configurable APR and term
- Automatic draw in 2016 if EBITDA is negative
- Annual debt service (interest + principal)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Key Inputs (Sidebar)

**Kit Sales Revenue**
- Base Kit Price (2016, $k): Starting price per kit
- Annual Price Escalation (%): Yearly price increase
- Base COGS per Kit (2016, $k): Starting cost per kit
- Annual COGS Escalation (%): Yearly COGS increase

**Kit Sales Forecast (2026-2035)**
- Individual sliders for each year 2026-2030
- Single slider for 2031-2035 (applied to all years)

**Engineering Services Revenue**
- Billing Rate ($/hr): Revenue per billable hour
- Cost per Billable Hour ($/hr): Direct labor cost
- Engineering Overhead (%): Overhead applied to direct costs

**Engineering Hours Forecast (2026-2035)**
- Individual sliders for each year 2026-2030 (in thousands)
- Single slider for 2031-2035 (applied to all years)

**Financial**
- Max Debt Available ($M): Debt facility cap
- Debt APR (%): Annual interest rate
- Debt Term (Years): Amortization period
- Income Tax Rate (%): Applied to positive taxable income
- WACC (%): Discount rate for DCF
- Terminal Growth Rate (%): Perpetual growth assumption

## Sensitivity Study Usage

1. Switch to "Sensitivity Study" mode using the radio button
2. Select 3 drivers from dropdowns (e.g., "2030 Units", "Base Kit Price", "2030 Eng Hours")
3. Choose an output metric (e.g., "Enterprise Value ($M)")
4. Configure ranges and number of points for each driver
5. Click "Run Sensitivity Study"
6. Review heatmap slices showing how driver combinations affect the metric

**Example Use Case**: "What combination of 2030 kit sales and engineering hours gets us to $500M enterprise value at different price points?"

## Model Assumptions

- **OpEx Schedule**: Hardcoded by year (2016: $2M → 2026-2035: $5-8M/year)
- **No Equity**: Model assumes debt-only financing (no equity raises)
- **Tax Floor**: No tax benefit for losses (taxes = max(0, taxable income) × tax rate)
- **Debt Draw**: Automatic in 2016 if EBITDA < 0, up to facility cap
- **Historical Period**: 2016-2025 (10 years)
- **Forecast Period**: 2026-2035 (10 years)

## Files

- `app.py` - Main Streamlit application (standalone model)
- `app_sense.py` - Sensitivity study module (3-driver analysis)
- `requirements.txt` - Python dependencies
- `A320Tam.jpg`, `logo.png` - Brand assets

## Repository

https://github.com/volocchio/Tamarack_525_Financials

## License

Proprietary - Tamarack Aerospace Group
