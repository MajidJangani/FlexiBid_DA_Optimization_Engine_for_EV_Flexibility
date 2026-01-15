# EV Fleet Flexibility Bidding Engine for DSO Markets

## Overview

An automated quantitative bidding system for electric vehicle fleet aggregators participating in UK Distribution System Operator (DSO) flexibility markets. The project optimizes day-ahead capacity offers while managing operational constraints, penalty risks, and competitive pricing dynamics.

**Market Context:** UK Power Networks (UKPN) pays EV fleets £130-215/vehicle/year to reduce charging during evening peak constraints (17:00-20:00), offering an alternative to costly substation upgrades (£1-5M, 3-7 years delivery).

**Business Challenge:** Successful participation requires submitting accurate 24-hour schedules, forecasting baselines with 95%+ accuracy to avoid penalties, pricing competitively against established aggregators, and maintaining driver trust—all simultaneously.

---

## Key Results

- **Revenue:** £138/vehicle/year risk-adjusted expected value (88% validation accuracy vs. operational trials)
- **Optimization:** 51% peak reduction capability while guaranteeing vehicle operational readiness
- **Forecasting:** 91% baseline accuracy avoiding Schedule Accuracy Factor (SAF) penalties
- **Pricing:** £436/MWh optimal bid (+6% premium over £410/MWh market leader for superior reliability)
- **Risk Model:** 192-scenario Monte Carlo analysis identifying event frequency (weather-driven, ±50% variance) and fleet participation (behavioral, 60-93% range) as dominant uncertainties

---

## Technical Approach

### **1. Market Intelligence (Module 00)**
Analyzed 18 months of UKPN dispatch data (14,813 events, May 2024 - Dec 2025):
- Market leader (47% share): £410/MWh average across 2,667 events
- Competitive landscape: 8 active aggregators across 88 flexibility zones
- Product focus: Day-Ahead Scheduled Utilisation (£439/MWh average, highest margin)

### **2. Fleet Simulation (Modules 01-03)**
Generated realistic Return-to-Home (R2H) commercial fleet behavior:
- Plug-in times: 17:00-20:00 (80% by 18:30)
- Energy requirements: 8.4-24.7 kWh/vehicle/night
- Charge point capacity: 7.4 kW single-phase domestic connections
- Behavioral opt-out rate: 5-10% modeled conservatively at 10%

### **3. Baseline Forecasting (Module 04)**
Predicted unmanaged charging demand for settlement and penalty calculations:
- Method: Historical unmanaged peak charging profiles (17:00-20:00)
- Accuracy: 95%+ required to avoid SAF penalties
- Output: 48 half-hour period schedule serving as bid baseline

### **4. Capacity Optimization (Module 05)**
Mixed-Integer Linear Programming (MILP) via Pyomo + GLPK:
- **Objective:** Maximize kW turn-down during constraint windows
- **Hard Constraints:** Energy delivery guarantees, charge point limits (1.4-7.4 kW), connection windows
- **Soft Constraints:** Rebound prevention (112% cap), minimum peak charging (25% baseline)
- **Output:** Optimized charging schedule delivering 51% peak reduction
Monte Carlo analysis across 192 scenarios (4×4×3×4 grid):
- **Grid conditions:** Event frequency (20-60 events/year, weather-driven)
- **Fleet participation:** Device uptime + driver opt-outs (60-93% effective availability)
- **Market competition:** Pricing pressure (£370-502/MWh range)
- **Forecasting accuracy:** SAF penalty exposure (85-95%+ accuracy via ML)

**Output:** Risk-adjusted expected revenue £138/vehicle (vs. £149 deterministic), with downside (5th percentile) £32/vehicle and upside (95th percentile) £291/vehicle.

**Pricing Strategy**
Market-based competitive pricing balancing win rate vs. margins:
- Target: 70-80% win rate (2-3× activation frequency vs. £549/MWh crisis pricing)
- Positioning: £436/MWh (+6% premium over market leader)
- Justification: Superior delivery reliability (97% vs. 90-95% market baseline)

**6. Risk Quantification**
Monte Carlo analysis across 192 scenarios (4×4×3×4 grid):
- **Grid conditions:** Event frequency (20-60 events/year, weather-driven)
- **Fleet participation:** Device uptime + driver opt-outs (60-93% effective availability)
- **Market competition:** Pricing pressure (£370-502/MWh range)
- **Forecasting accuracy:** SAF penalty exposure (85-95%+ accuracy via ML)

**Output:** Risk-adjusted expected revenue £138/vehicle (vs. £149 deterministic), with downside (5th percentile) £32/vehicle and upside (95th percentile) £291/vehicle.

---

## International Transferability

The modular architecture enables systematic geographic expansion. **Sweden Case Study (Effekthandel Väst)** demonstrates:

- **Baseline methodology dominates business viability:** Asset capacity baselines (MaxUsage product: 28% cost reduction) outperform historic demand baselines (ShortFlex: 1-2%) by **14× revenue multiplier**
- **V2G value is market-dependent:** Bidirectional charging increases revenue 300-500% in historic baseline markets vs. 15% in technology profile markets
- **Product diversity enables risk management:** Combining passive products (MaxUsage) with activation products (LongFlex) creates portfolio diversification impossible in single-product markets

**Adaptation Timeline:** 6 weeks to port framework from UK to Sweden (only market-interface modules require localization, core optimization engine unchanged).

---

## Data Sources

- **UKPN Dispatch Records:** 14,813 events, 18 months (May 2024 - Dec 2025)
- **Fleet Behavior:** Centrica WS1 trial (65 R2H EVs, 60+ flexibility events, 2017-2021)
- **Tariff Data:** Octopus Energy Agile API (live spot prices)
- **Vehicle Specifications:** UK EV Database (make/model/battery/efficiency)
- **Validation Benchmark:** WS1 trial outcomes (£172/vehicle/year crisis-year baseline)

---

## Repository Structure
```
├── module_00_market_analysis.py          # UKPN data scraping & competitive benchmarking
├── module_01_data_ingestion.py          # Fleet telematics & tariff data integration
├── module_02_fleet_generator.py         # Stochastic R2H fleet simulation
├── module_03_operational_requirements.py # Vehicle readiness constraints
├── module_04_baseline_forecasting.py    # Unmanaged charging prediction
├── module_05_flexibility_optimization_engine.py # MILP capacity optimization
├── module_06_bid_output.py              # API submission formatting
├── module_07_pricing_revenue.py         # Market-based pricing strategy
├── module_08_validation.py              # WS1 benchmark comparison
├── module_09_scenario_analysis_risk_based.py # Monte Carlo risk quantification
└── international_expansion/
    └── sweden_effekthandel_vast.py      # Market adaptation framework
```

---

## Commercial Applications

1. **Operational Bidding Automation:** Generate daily bids (48-PTU schedule + capacity + price) for live market participation
2. **Market Entry Feasibility:** Assess new geographies (Netherlands GOPACS, Norway NorFlex, Sweden) by comparing revenue potential, technical requirements, and risk profiles
3. **DER Portfolio Optimization:** Extend beyond EVs to batteries, heat pumps, industrial demand response with multi-market revenue stacking

---

## Technology Stack

- **Optimization:** Pyomo (MILP modeling), GLPK (solver)
- **Data Processing:** Pandas, NumPy
- **Risk Analysis:** Monte Carlo simulation (NumPy random sampling)
- **Visualization:** Matplotlib, Seaborn
- **API Integration:** Requests (Octopus Agile, UKPN platform)

---

## Validation & Limitations

**Validation Score:** 88.2/100 against WS1 trial benchmarks
- ✅ Revenue: £138/vehicle (model) vs. £172/vehicle (WS1 trial) — 15% variance within operational range
- ✅ Peak reduction: 51.1% (model) vs. 48.3% (WS1 trial) — Improved optimization
- ✅ Forecast accuracy: 91% (Day-Ahead model) vs. 80% (WS1 Month-Ahead) — Product difference

**Limitations:**
- Revenue projections validated for strategic decision-making, not contract performance guarantees (requires 6-12 month operational validation)
- Behavioral opt-out rates modeled at 10% conservatively; actual rates vary 5-15% by fleet type
- V2G hardware costs (£800/vehicle) not included in baseline model (optional extension)

---

## License & Attribution

This project is developed for academic and commercial analysis purposes. Market data sourced from publicly available UKPN dispatch records. Fleet behavior calibrated to published WS1 trial outcomes (Centrica, 2017-2021).

**Contact:** [Your GitHub/Email]

---

## Citation

If using this framework in research or commercial applications:
```
Jangani, M. (2025). EV Fleet Flexibility Bidding Engine for DSO Markets. 
GitHub: [repository_url]
```