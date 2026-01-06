"""
COMPLETE PIPELINE TEST
======================
Run Module 05 → Generate outputs → Run Scenario Analysis → Compare results

Save this as: test_complete_pipeline.py
Run with: python test_complete_pipeline.py
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

print("="*80)
print("🧪 COMPLETE PIPELINE TEST")
print("="*80)

# ============================================================================
# STEP 1: RUN MODULE 05 (Optimization Engine)
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 1: RUNNING MODULE 05 OPTIMIZATION")
print("="*80)

try:
    # Import Module 05
    from module_05_flexibility_optimization_engine import FlexibilityBiddingEngine
    
    # Initialize engine
    engine = FlexibilityBiddingEngine(
        operational_csv='data/operational_constraints.csv',
        flexible_units_csv='data/flexible_units.csv',
        baseline_csv='data/baseline_profile.csv',
        ukpn_market_csv='data/ukpnflexibilitydemandturndown.csv'
    )
    
    # Run complete optimization
    print("\n🚀 Running optimization pipeline...")
    module05_results = engine.execute_complete_pipeline(
        solver='glpk',
        save_outputs=True,
        optimization_mode='flexibility'
    )
    
    print("\n✅ Module 05 completed successfully!")
    
    # Extract key metrics
    business_case = module05_results['business_case']
    
    print("\n📊 MODULE 05 RESULTS:")
    print(f"   Fleet Size: {business_case['fleet_size']} vehicles")
    print(f"   Total Capacity: {business_case['total_capacity_kw']:.1f} kW")
    print(f"   Revenue/Vehicle: £{business_case['revenue_per_vehicle_annual']:.0f}/year")
    print(f"   vs WS1: {business_case['vs_ws1_benchmark_pct']:+.1f}%")
    print(f"   Average Price: £{business_case['avg_price_gbp_mwh']:.0f}/MWh")
    
except Exception as e:
    print(f"\n❌ Module 05 failed: {e}")
    print("\nFalling back to example data...")
    
    # Create example business case
    business_case = {
        'fleet_size': 65,
        'num_fus': 17,
        'total_capacity_kw': 210.2,
        'avg_price_gbp_mwh': 441.0,
        'total_annual_revenue': 9620.0,
        'revenue_per_vehicle_annual': 148.0,
        'vs_ws1_benchmark_pct': -31.2,
        'avg_delivery_confidence': 0.95
    }
    
    # Save example business case
    os.makedirs('data', exist_ok=True)
    with open('data/business_case_summary.json', 'w') as f:
        json.dump({'business_case': business_case}, f, indent=2)
    
    print("✅ Created example business case")


# ============================================================================
# STEP 2: RUN SCENARIO ANALYSIS (Risk-Based)
# ============================================================================

print("\n" + "="*80)
print("🎯 STEP 2: RUNNING SCENARIO ANALYSIS")
print("="*80)

try:
    from scenario_analysis_risk_based import RiskScenarioAnalysis
    
    # Initialize scenario engine
    scenario_engine = RiskScenarioAnalysis(
        baseline_results_path='data/business_case_summary.json'
    )
    
    # Run complete analysis
    print("\n⚙️  Generating 45 risk scenarios...")
    scenario_results = scenario_engine.run_complete_analysis()
    
    print("\n✅ Scenario analysis completed!")
    
    # Extract key metrics
    risk_metrics = scenario_results['risk_metrics']
    
    print("\n📊 SCENARIO ANALYSIS RESULTS:")
    print(f"   Expected Value: £{risk_metrics['expected_value']:.0f}/vehicle")
    print(f"   vs Baseline: {risk_metrics['ev_vs_baseline_pct']:+.1f}%")
    print(f"   5th Percentile: £{risk_metrics['var_5_percentile']:.0f}/vehicle")
    print(f"   95th Percentile: £{risk_metrics['upside_95_percentile']:.0f}/vehicle")
    print(f"   Std Deviation: £{risk_metrics['std_dev']:.0f}/vehicle")
    
except Exception as e:
    print(f"\n❌ Scenario analysis failed: {e}")
    risk_metrics = None
    scenario_results = None


# ============================================================================
# STEP 3: COMPARE RESULTS (Module 05 vs Risk-Adjusted)
# ============================================================================

print("\n" + "="*80)
print("📈 STEP 3: COMPARING RESULTS")
print("="*80)

comparison_data = []

# Module 05 baseline
comparison_data.append({
    'Metric': 'Module 05 (Deterministic)',
    'Revenue/Vehicle': f"£{business_case['revenue_per_vehicle_annual']:.0f}",
    'Total Revenue': f"£{business_case['total_annual_revenue']:.0f}",
    'vs WS1': f"{business_case['vs_ws1_benchmark_pct']:+.1f}%",
    'Notes': 'Optimistic - assumes no variance'
})

# Scenario analysis (if available)
if risk_metrics:
    comparison_data.append({
        'Metric': 'Scenario Analysis (Expected Value)',
        'Revenue/Vehicle': f"£{risk_metrics['expected_value']:.0f}",
        'Total Revenue': f"£{risk_metrics['expected_value'] * business_case['fleet_size']:.0f}",
        'vs WS1': f"{(risk_metrics['expected_value'] / 215 - 1) * 100:+.1f}%",
        'Notes': 'Risk-adjusted - probability-weighted'
    })
    
    comparison_data.append({
        'Metric': 'Downside (5th Percentile)',
        'Revenue/Vehicle': f"£{risk_metrics['var_5_percentile']:.0f}",
        'Total Revenue': f"£{risk_metrics['var_5_percentile'] * business_case['fleet_size']:.0f}",
        'vs WS1': f"{(risk_metrics['var_5_percentile'] / 215 - 1) * 100:+.1f}%",
        'Notes': 'Conservative - bad scenario'
    })
    
    comparison_data.append({
        'Metric': 'Upside (95th Percentile)',
        'Revenue/Vehicle': f"£{risk_metrics['upside_95_percentile']:.0f}",
        'Total Revenue': f"£{risk_metrics['upside_95_percentile'] * business_case['fleet_size']:.0f}",
        'vs WS1': f"{(risk_metrics['upside_95_percentile'] / 215 - 1) * 100:+.1f}%",
        'Notes': 'Optimistic - good scenario'
    })

# WS1 Benchmark
comparison_data.append({
    'Metric': 'WS1 Trial (Benchmark)',
    'Revenue/Vehicle': '£215',
    'Total Revenue': f"£{215 * business_case['fleet_size']:.0f}",
    'vs WS1': '0.0%',
    'Notes': 'British Gas trial (60 events, crisis year)'
})

comparison_df = pd.DataFrame(comparison_data)

print("\n" + comparison_df.to_string(index=False))


# ============================================================================
# STEP 4: VALIDATE PRICING STRATEGY
# ============================================================================

print("\n" + "="*80)
print("💰 STEP 4: PRICING VALIDATION")
print("="*80)

# Load FU bids (if available)
try:
    fu_bids = pd.read_csv('data/fu_bids_day_ahead.csv')
    
    print("\n📊 FU BID SUMMARY:")
    print(f"   Total FUs: {len(fu_bids)}")
    print(f"   Avg Price: £{fu_bids['price_gbp_mwh'].mean():.0f}/MWh")
    print(f"   Price Range: £{fu_bids['price_gbp_mwh'].min():.0f} - £{fu_bids['price_gbp_mwh'].max():.0f}/MWh")
    
    # Pricing strategy distribution
    print("\n📋 PRICING STRATEGIES:")
    strategy_counts = fu_bids['pricing_strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        pct = count / len(fu_bids) * 100
        print(f"   {strategy:30s} {count:>3d} FUs ({pct:>5.1f}%)")
    
    # Compare to market (Axle = £410/MWh)
    axle_price = 410
    our_avg = fu_bids['price_gbp_mwh'].mean()
    price_diff = (our_avg / axle_price - 1) * 100
    
    print(f"\n🎯 COMPETITIVENESS:")
    print(f"   Axle Energy (market leader): £{axle_price}/MWh")
    print(f"   Our average bid: £{our_avg:.0f}/MWh")
    print(f"   Difference: {price_diff:+.1f}%")
    
    if abs(price_diff) < 10:
        print(f"   ✅ COMPETITIVE: Within 10% of market leader")
    elif abs(price_diff) < 20:
        print(f"   ⚠️  ACCEPTABLE: Within 20% of market leader")
    else:
        print(f"   ❌ RISK: >20% difference may hurt win rate")
    
    # Save pricing analysis
    pricing_summary = {
        'avg_price': float(our_avg),
        'vs_axle_pct': float(price_diff),
        'num_fus': len(fu_bids),
        'strategy_distribution': strategy_counts.to_dict(),
        'price_range': {
            'min': float(fu_bids['price_gbp_mwh'].min()),
            'max': float(fu_bids['price_gbp_mwh'].max())
        }
    }
    
    with open('outputs/pricing_analysis.json', 'w') as f:
        json.dump(pricing_summary, f, indent=2)
    
    print("\n💾 Saved pricing analysis to outputs/pricing_analysis.json")
    
except Exception as e:
    print(f"⚠️  Could not load FU bids: {e}")


# ============================================================================
# STEP 5: CHECK OUTPUT FILES
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 5: OUTPUT FILES CHECK")
print("="*80)

expected_outputs = {
    'Module 05 Outputs': [
        'data/fu_bids_day_ahead.csv',
        'data/business_case_summary.json',
        'outputs/portfolio_visualization.png'
    ],
    'Scenario Analysis Outputs': [
        'data/risk_scenario_results.csv',
        'data/risk_metrics_summary.json',
        'outputs/risk_scenario_analysis.png',
        'outputs/scenario_comparison_table.csv'
    ]
}

print("\n✅ FILE CHECK:")
for category, files in expected_outputs.items():
    print(f"\n{category}:")
    for filepath in files:
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   ✅ [{size_kb:>7.1f} KB]  {filepath}")
        else:
            print(f"   ❌ [MISSING]      {filepath}")


# ============================================================================
# STEP 6: EXPORT COMPREHENSIVE COMPARISON CSV
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 6: EXPORTING COMPARISON DATA")
print("="*80)

# Create comprehensive comparison
os.makedirs('outputs', exist_ok=True)

# Save comparison table
comparison_df.to_csv('outputs/revenue_comparison.csv', index=False)
print("\n✅ Saved: outputs/revenue_comparison.csv")

# Create detailed metrics CSV
if risk_metrics:
    detailed_metrics = {
        'Metric': [
            'Module 05 Revenue/Vehicle',
            'Expected Value (Risk-Adjusted)',
            'Worst Case (5th %ile)',
            'Best Case (95th %ile)',
            'Standard Deviation',
            'Coefficient of Variation',
            'Average Price',
            'Fleet Size',
            'Total Capacity'
        ],
        'Value': [
            f"£{business_case['revenue_per_vehicle_annual']:.0f}",
            f"£{risk_metrics['expected_value']:.0f}",
            f"£{risk_metrics['var_5_percentile']:.0f}",
            f"£{risk_metrics['upside_95_percentile']:.0f}",
            f"£{risk_metrics['std_dev']:.0f}",
            f"{risk_metrics['coefficient_of_variation']:.2f}",
            f"£{business_case['avg_price_gbp_mwh']:.0f}/MWh",
            f"{business_case['fleet_size']} vehicles",
            f"{business_case['total_capacity_kw']:.1f} kW"
        ],
        'Notes': [
            'Deterministic model',
            'Probability-weighted mean',
            'Downside risk (VaR)',
            'Upside potential',
            'Revenue volatility',
            'Risk measure (σ/μ)',
            'Average bid price',
            'Participating vehicles',
            'Total turn-down capacity'
        ]
    }
    
    detailed_df = pd.DataFrame(detailed_metrics)
    detailed_df.to_csv('outputs/detailed_metrics.csv', index=False)
    print("✅ Saved: outputs/detailed_metrics.csv")


# ============================================================================
# STEP 7: FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ COMPLETE PIPELINE TEST FINISHED")
print("="*80)

ws1_target = 215

print(f"""
📊 FINAL SUMMARY:

MODULE 05 (Deterministic):
  Revenue/Vehicle:     £{business_case['revenue_per_vehicle_annual']:.0f}/year
  vs WS1 Target:       {business_case['vs_ws1_benchmark_pct']:+.1f}%
  Average Price:       £{business_case['avg_price_gbp_mwh']:.0f}/MWh
""")

if risk_metrics:
    print(f"""SCENARIO ANALYSIS (Risk-Adjusted):
  Expected Value:      £{risk_metrics['expected_value']:.0f}/year ({risk_metrics['ev_vs_baseline_pct']:+.1f}% vs baseline)
  Downside (5th):      £{risk_metrics['var_5_percentile']:.0f}/year ({(risk_metrics['var_5_percentile']/215 - 1)*100:+.1f}% vs WS1)
  Upside (95th):       £{risk_metrics['upside_95_percentile']:.0f}/year ({(risk_metrics['upside_95_percentile']/215 - 1)*100:+.1f}% vs WS1)
  Volatility (σ):      £{risk_metrics['std_dev']:.0f}/year
""")

print(f"""VALIDATION:
  WS1 Target:          £{ws1_target}/year
  Gap to Target:       £{business_case['revenue_per_vehicle_annual'] - ws1_target:.0f}/year
  % of Target:         {business_case['revenue_per_vehicle_annual'] / ws1_target * 100:.1f}%
""")

# Recommendations
print("🎯 KEY INSIGHTS:")

if business_case['revenue_per_vehicle_annual'] < ws1_target * 0.80:
    print("  ⚠️  Revenue significantly below WS1 target")
    print("     → Check pricing strategy (too conservative?)")
    print("     → Review SAF penalties (too aggressive reduction?)")
    print("     → Consider increasing event frequency assumption")
elif business_case['revenue_per_vehicle_annual'] > ws1_target * 1.20:
    print("  ⚠️  Revenue significantly above WS1 target")
    print("     → Model may be too optimistic")
    print("     → Review SAF assumptions")
    print("     → Check event duration (1.5h correct?)")
else:
    print("  ✅ Revenue within reasonable range of WS1 target")
    print(f"     → Gap: {abs(business_case['revenue_per_vehicle_annual'] - ws1_target):.0f} ({abs((business_case['revenue_per_vehicle_annual'] / ws1_target - 1) * 100):.1f}%)")

if risk_metrics and risk_metrics['std_dev'] > 50:
    print(f"\n  ⚠️  High volatility (σ = £{risk_metrics['std_dev']:.0f})")
    print("     → Weather is primary risk driver")
    print("     → Consider zone diversification")
    print("     → Maintain 2-3× cash buffer")

print("\n📂 OUTPUT FILES:")
print("  • outputs/revenue_comparison.csv")
print("  • outputs/detailed_metrics.csv")
if os.path.exists('outputs/pricing_analysis.json'):
    print("  • outputs/pricing_analysis.json")
if os.path.exists('outputs/risk_scenario_analysis.png'):
    print("  • outputs/risk_scenario_analysis.png")

print("\n" + "="*80)