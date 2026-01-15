"""
RISK-BASED SCENARIO ANALYSIS ENGINE
====================================
Analyzes EV flexibility revenue under 4 critical business risks:
1. Grid Conditions (Event Frequency) - Uncontrollable
2. Device Performance (Uptime/Connectivity) - Partially controllable
3. Market Competition (Pricing Pressure) - Uncontrollable
4. Forecasting Accuracy (SAF Penalties) - Controllable

Author: Portfolio Project - EV Flexibility Bidding
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class RiskScenarioAnalysis:
    """
    Portfolio-grade scenario analysis focusing on real business risks.
    
    Unlike academic "what-if" analysis, this focuses on:
    - Risks you CANNOT control (grid, competition)
    - Risks you CAN partially control (device performance, forecasting)
    - Probability-weighted expected values
    - Business decision support
    """
    
    def __init__(self, baseline_results_path='data/business_case_summary.json'):
        """Initialize with baseline optimization results"""
        self.baseline_results = self.load_baseline_results(baseline_results_path)
        self.scenarios = {}
        self.combined_scenarios = {}
        
    def load_baseline_results(self, path):
        """Load baseline results from Module 05 optimization"""
        try:
            with open(path, 'r') as f:
                results = json.load(f)
            
            print("="*70)
            print("📊 RISK-BASED SCENARIO ANALYSIS ENGINE")
            print("="*70)
            print(f"\n✅ Loaded baseline results from: {path}")
            
            # Flexible key extraction - handle different JSON structures
            business_case = results.get('business_case', results)
            
            # Try different possible key names
            revenue_per_vehicle = (
                business_case.get('revenue_per_vehicle_annual') or
                business_case.get('revenue_per_vehicle') or
                business_case.get('annual_revenue_per_vehicle') or
                148.0  # Fallback
            )
            
            total_revenue = (
                business_case.get('total_annual_revenue') or
                business_case.get('total_revenue') or
                business_case.get('annual_total_revenue') or
                0.0
            )
            
            fleet_size = (
                business_case.get('fleet_size_participating') or
                business_case.get('fleet_size') or
                business_case.get('num_vehicles') or
                business_case.get('participating_vehicles') or
                65  # Fallback
            )
            
            avg_price = (
                business_case.get('avg_price_gbp_mwh') or
                business_case.get('average_price') or
                business_case.get('price_gbp_mwh') or
                441.0  # Fallback
            )
            
            capacity = (
                business_case.get('total_capacity_kw') or
                business_case.get('capacity_kw') or
                business_case.get('total_capacity') or
                0.0
            )
            
            print(f"   Revenue per vehicle: £{revenue_per_vehicle:.0f}/year")
            if total_revenue > 0:
                print(f"   Total fleet revenue: £{total_revenue:.0f}/year")
            print(f"   Fleet size: {fleet_size} vehicles")
            print(f"   Average price: £{avg_price:.0f}/MWh")
            if capacity > 0:
                print(f"   Total capacity: {capacity:.1f} kW")
            
            # Standardize the output format
            standardized_results = {
                'business_case': {
                    'revenue_per_vehicle_annual': float(revenue_per_vehicle),
                    'total_annual_revenue': float(total_revenue) if total_revenue > 0 else float(revenue_per_vehicle * fleet_size),
                    'fleet_size_participating': int(fleet_size),
                    'avg_price_gbp_mwh': float(avg_price),
                    'total_capacity_kw': float(capacity)
                }
            }
            
            return standardized_results
            
        except FileNotFoundError:
            print(f"⚠️  Baseline results not found at: {path}")
            print("   Using example baseline values...")
            
            # Example baseline for demonstration
            return {
                'business_case': {
                    'revenue_per_vehicle_annual': 148.0,
                    'total_annual_revenue': 9620.0,
                    'fleet_size_participating': 65,
                    'avg_price_gbp_mwh': 441.0,
                    'total_capacity_kw': 210.2
                }
            }
        except Exception as e:
            print(f"⚠️  Error loading baseline results: {e}")
            print("   Using example baseline values...")
            
            return {
                'business_case': {
                    'revenue_per_vehicle_annual': 148.0,
                    'total_annual_revenue': 9620.0,
                    'fleet_size_participating': 65,
                    'avg_price_gbp_mwh': 441.0,
                    'total_capacity_kw': 210.2
                }
            }
    
    def define_risk_scenarios(self):
        """
        Define 4 core risk dimensions based on real business threats.
        
        Risk 1: Grid Conditions (Event Frequency) - Uncontrollable
        Risk 2: Device Performance (Uptime) - Partially Controllable
        Risk 3: Market Competition (Pricing) - Uncontrollable
        Risk 4: Forecasting Accuracy (SAF Penalties) - Controllable
        """
        
        print("\n" + "="*70)
        print("🎯 DEFINING RISK SCENARIOS")
        print("="*70)
        
        # ====================================================================
        # RISK 1: GRID CONDITIONS (Event Frequency - Weather-Driven)
        # ====================================================================

        self.weather_scenarios = {
            'mild_winter_low_demand': {
                'events_per_year': 20,  # 50% of baseline
                'price_environment': 0.90,
                'revenue_multiplier': 0.45,  # (20/40) × 0.90
                'probability': 0.15,
                'description': 'Mild winter, high renewables, low heating demand',
                'driver': 'Weather-driven (uncontrollable)',
                'historical_precedent': 'Summer 2023 - mild conditions',
                'controllability': 'NONE - Weather dependent',
                'note': 'Represents 2× baseline deviation (40 → 20 events)'
            },
            
            'normal_winter': {
                'events_per_year': 40,  # Baseline
                'price_environment': 1.00,
                'revenue_multiplier': 1.00,
                'probability': 0.60,
                'description': 'Normal winter conditions',
                'driver': 'Baseline assumption',
                'historical_precedent': 'Winter 2023/24',
                'controllability': 'NONE - Weather dependent'
            },
            
            'harsh_winter_crisis': {
                'events_per_year': 60,  # 150% of baseline
                'events_range': '50-70 (plausible estimate)',
                'price_environment': 1.30,  # Supply constraints
                'revenue_multiplier': 1.95,  # (60/40) × 1.30
                'probability': 0.20,
                'description': 'Cold snap, low wind generation, gas supply issues',
                'driver': 'Weather + supply constraints (uncontrollable)',
                'historical_precedent': 'Winter 2021/22 - Energy crisis',
                'controllability': 'NONE - Weather dependent',
                'note': 'Probability increased to 0.20'
            },
            
            'mild_summer': {
                'events_per_year': 10,  # Very low demand
                'price_environment': 0.85,
                'revenue_multiplier': 0.21,  # (10/40) × 0.85
                'probability': 0.05,
                'description': 'Summer months, minimal heating demand',
                'driver': 'Seasonal variation',
                'historical_precedent': 'Typical summer baseline',
                'controllability': 'NONE - Seasonal pattern'
            }
        }
        
        print(f"\n1. Grid Conditions (Event Frequency Risk):")
        print(f"   Weather scenarios: {len(self.weather_scenarios)}")
        
        # ====================================================================
        # RISK 2: DEVICE PERFORMANCE (Uptime/Connectivity Risk)
        # ====================================================================
        
        self.device_scenarios = {
            'high_uptime': {
                'participation_rate': 0.93,
                'revenue_multiplier': 1.033,  # 0.93/0.90
                'probability': 0.30,
                'description': 'Excellent connectivity, stable OEM platform',
                'driver': 'Hypervolt/GivEnergy API uptime 99%+',
                'controllability': 'MEDIUM - OEM partnership SLAs',
                'mitigation': '99% uptime SLA with device manufacturers'
            },
            
            'baseline_uptime': {
                'participation_rate': 0.90,
                'revenue_multiplier': 1.00,
                'probability': 0.50,
                'description': 'Normal device operations',
                'driver': 'Typical API/connectivity performance',
                'controllability': 'MEDIUM - Standard operations',
                'mitigation': 'Monitoring and alerting systems'
            },
            
            'degraded_uptime': {
                'participation_rate': 0.80,
                'revenue_multiplier': 0.889,  # 0.80/0.90
                'probability': 0.15,
                'description': 'API issues, firmware bugs',
                'driver': 'OEM platform instability',
                'controllability': 'MEDIUM - Escalate with OEM',
                'mitigation': 'Redundant device integrations, fallback systems'
            },
            
            'critical_failure': {
                'participation_rate': 0.60,
                'revenue_multiplier': 0.667,  # 0.60/0.90
                'probability': 0.05,
                'description': 'Major OEM platform outage',
                'driver': 'Hypervolt/GivEnergy system down',
                'controllability': 'MEDIUM - Partner diversification',
                'mitigation': 'Multi-OEM integration strategy'
            }
        }
        
        print(f"2. Device Performance (Uptime Risk):")
        print(f"   Device scenarios: {len(self.device_scenarios)}")
        
        # ====================================================================
        # RISK 3: MARKET COMPETITION (Pricing Pressure)
        # ====================================================================
        
        self.competition_scenarios = {
            'low_competition': {
                'aggregators_active': 2,
                'price_premium': 1.15,
                'revenue_multiplier': 1.15,
                'probability': 0.20,
                'description': 'Few aggregators, limited capacity, premium pricing',
                'driver': 'Market undersupply',
                'controllability': 'NONE - Market dynamics',
                'mitigation': 'Lock in long-term contracts during favorable periods'
            },
            
            'normal_competition': {
                'aggregators_active': 4,
                'price_premium': 1.00,
                'revenue_multiplier': 1.00,
                'probability': 0.60,
                'description': 'Balanced market, fair pricing',
                'driver': 'Competitive equilibrium',
                'controllability': 'NONE - Market dynamics',
                'mitigation': 'Maintain competitive edge through reliability'
            },
            
            'high_competition': {
                'aggregators_active': 8,
                'price_premium': 0.85,
                'revenue_multiplier': 0.85,
                'probability': 0.20,
                'description': 'Market saturation, price wars, margin compression',
                'driver': 'New entrants, oversupply',
                'controllability': 'NONE - Market dynamics',
                'mitigation': 'Differentiate through service quality'
            }
        }
        
        print(f"3. Market Competition (Pricing Risk):")
        print(f"   Competition scenarios: {len(self.competition_scenarios)}")
        
        # ====================================================================
        # RISK 4: FORECASTING ACCURACY (SAF Penalties)
        # ====================================================================
        
        self.forecasting_scenarios = {
            'excellent_accuracy': {
                'saf_multiplier': 1.00,
                'accuracy_pct': 95,
                'revenue_multiplier': 1.00,
                'probability': 0.40,
                'description': '95%+ accuracy, no SAF penalty',
                'driver': 'Advanced ML forecasting, mature operations',
                'controllability': 'HIGH - Invest in forecasting systems',
                'mitigation': 'ML-based demand prediction, conservative buffers'
            },
            
            'good_accuracy': {
                'saf_multiplier': 0.90,
                'accuracy_pct': 90,
                'revenue_multiplier': 0.90,
                'probability': 0.40,
                'description': '90% accuracy, 10% SAF penalty',
                'driver': 'Standard forecasting methods',
                'controllability': 'HIGH - Controllable through investment',
                'mitigation': 'Baseline forecasting improvements'
            },
            
            'poor_accuracy': {
                'saf_multiplier': 0.70,
                'accuracy_pct': 85,
                'revenue_multiplier': 0.70,
                'probability': 0.15,
                'description': '85% accuracy, 30% SAF penalty',
                'driver': 'Baseline errors, new market entry',
                'controllability': 'HIGH - Learning curve issue',
                'mitigation': 'Experience + system refinement'
            },
            
            'critical_miss': {
                'saf_multiplier': 0.40,
                'accuracy_pct': 75,
                'revenue_multiplier': 0.40,
                'probability': 0.05,
                'description': '<80% accuracy, severe SAF penalty',
                'driver': 'Major forecasting failure',
                'controllability': 'HIGH - Preventable with proper systems',
                'mitigation': 'Redundant forecasting models, real-time monitoring'
            }
        }
        
        print(f"4. Forecasting Accuracy (SAF Penalty Risk):")
        print(f"   Forecasting scenarios: {len(self.forecasting_scenarios)}")
        
        total_combinations = (len(self.weather_scenarios) * 
                            len(self.device_scenarios) * 
                            len(self.competition_scenarios) * 
                            len(self.forecasting_scenarios))
        
        print(f"\n   Total combinations: {total_combinations}")
        
        return {
            'weather': self.weather_scenarios,
            'device': self.device_scenarios,
            'competition': self.competition_scenarios,
            'forecasting': self.forecasting_scenarios
        }
    
    def calculate_combined_scenarios(self):
        """
        Calculate all combinations of risk scenarios.
        
        Returns DataFrame with:
        - Scenario ID
        - Combined probability
        - Revenue per vehicle
        - Risk factor details
        """
        
        print("\n" + "="*70)
        print("🔄 CALCULATING COMBINED SCENARIOS")
        print("="*70)
        
        baseline_revenue = self.baseline_results['business_case']['revenue_per_vehicle_annual']
        
        scenarios = []
        
        # Generate all combinations
        for weather_name, weather in self.weather_scenarios.items():
            for device_name, device in self.device_scenarios.items():
                for comp_name, comp in self.competition_scenarios.items():
                    for forecast_name, forecast in self.forecasting_scenarios.items():
                        
                        scenario_id = f"{weather_name}_{device_name}_{comp_name}_{forecast_name}"
                        
                        # Combined probability
                        combined_prob = (weather['probability'] *
                                       device['probability'] *
                                       comp['probability'] *
                                       forecast['probability'])
                        
                        # Combined revenue multiplier (multiplicative)
                        combined_multiplier = (weather['revenue_multiplier'] *
                                             device['revenue_multiplier'] *
                                             comp['revenue_multiplier'] *
                                             forecast['revenue_multiplier'])
                        
                        # Calculate scenario revenue
                        scenario_revenue = baseline_revenue * combined_multiplier
                        
                        scenarios.append({
                            'scenario_id': scenario_id,
                            'weather': weather_name,
                            'device': device_name,
                            'competition': comp_name,
                            'forecasting': forecast_name,
                            'probability': combined_prob,
                            'revenue_multiplier': combined_multiplier,
                            'revenue_per_vehicle': scenario_revenue,
                            'weather_events': weather.get('events_per_year', 40),
                            'grid_events': weather.get('events_per_year', 40),  # Alias
                            'device_uptime': device.get('participation_rate', 0.90),
                            'price_multiplier': comp['revenue_multiplier'],
                            'saf_multiplier': forecast['saf_multiplier']
                        })
        
        # Create DataFrame
        self.scenario_df = pd.DataFrame(scenarios)
        
        # Sort by probability (most likely first)
        self.scenario_df = self.scenario_df.sort_values('probability', ascending=False).reset_index(drop=True)
        
        print(f"   ✅ Generated {len(self.scenario_df)} scenarios")
        print(f"   Probability sum: {self.scenario_df['probability'].sum():.4f}")
        
        return self.scenario_df
    
    def calculate_expected_value(self):
        """
        Calculate probability-weighted expected value and risk metrics.
        """
        
        print("\n" + "="*70)
        print("📊 CALCULATING RISK METRICS")
        print("="*70)
        
        baseline_revenue = self.baseline_results['business_case']['revenue_per_vehicle_annual']
        
        # Expected value (probability-weighted mean)
        expected_value = (self.scenario_df['revenue_per_vehicle'] * 
                         self.scenario_df['probability']).sum()
        
        # Percentiles (using cumulative probability)
        sorted_scenarios = self.scenario_df.sort_values('revenue_per_vehicle')
        cumulative_prob = sorted_scenarios['probability'].cumsum()
        
        var_5 = sorted_scenarios[cumulative_prob >= 0.05].iloc[0]['revenue_per_vehicle']
        var_10 = sorted_scenarios[cumulative_prob >= 0.10].iloc[0]['revenue_per_vehicle']
        median = sorted_scenarios[cumulative_prob >= 0.50].iloc[0]['revenue_per_vehicle']
        upside_90 = sorted_scenarios[cumulative_prob >= 0.90].iloc[0]['revenue_per_vehicle']
        upside_95 = sorted_scenarios[cumulative_prob >= 0.95].iloc[0]['revenue_per_vehicle']
        
        worst_case = self.scenario_df['revenue_per_vehicle'].min()
        best_case = self.scenario_df['revenue_per_vehicle'].max()
        
        # Volatility metrics
        variance = ((self.scenario_df['revenue_per_vehicle'] - expected_value) ** 2 * 
                   self.scenario_df['probability']).sum()
        std_dev = np.sqrt(variance)
        
        self.risk_metrics = {
            'baseline_revenue': baseline_revenue,
            'expected_value': expected_value,
            'ev_vs_baseline_pct': (expected_value / baseline_revenue - 1) * 100,
            'var_5_percentile': var_5,
            'var_10_percentile': var_10,
            'median': median,
            'upside_90_percentile': upside_90,
            'upside_95_percentile': upside_95,
            'worst_case': worst_case,
            'best_case': best_case,
            'std_dev': std_dev,
            'coefficient_of_variation': std_dev / expected_value if expected_value > 0 else 0,
            'range_multiplier': best_case / worst_case if worst_case > 0 else 0
        }
        
        print(f"\n💰 EXPECTED VALUE ANALYSIS:")
        print(f"   Baseline:          £{baseline_revenue:.0f}/vehicle")
        print(f"   Expected Value:    £{expected_value:.0f}/vehicle ({(expected_value/baseline_revenue-1)*100:+.1f}%)")
        print(f"   Median:            £{median:.0f}/vehicle")
        
        print(f"\n📉 RISK EXPOSURE:")
        print(f"   Worst Case:        £{worst_case:.0f}/vehicle ({worst_case/baseline_revenue*100:.0f}% of baseline)")
        print(f"   5th Percentile:    £{var_5:.0f}/vehicle ({var_5/baseline_revenue*100:.0f}% of baseline)")
        print(f"   95th Percentile:   £{upside_95:.0f}/vehicle ({upside_95/baseline_revenue*100:.0f}% of baseline)")
        print(f"   Best Case:         £{best_case:.0f}/vehicle ({best_case/baseline_revenue*100:.0f}% of baseline)")
        
        print(f"\n📊 VOLATILITY:")
        print(f"   Standard Dev:      £{std_dev:.0f}/vehicle")
        print(f"   Coeff. of Var:     {std_dev/expected_value:.2f}")
        print(f"   Range (×):         {best_case/worst_case:.1f}×")
        
        return self.risk_metrics
    
    def identify_key_scenarios(self):
        """
        Identify and categorize key scenarios for decision-making.
        """
        
        print("\n" + "="*70)
        print("🎯 IDENTIFYING KEY SCENARIOS")
        print("="*70)
        
        # Most likely scenario
        most_likely = self.scenario_df.iloc[0]
        
        # Best case scenario
        best_case_scenario = self.scenario_df.loc[
            self.scenario_df['revenue_per_vehicle'].idxmax()
        ]
        
        # Worst case scenario
        worst_case_scenario = self.scenario_df.loc[
            self.scenario_df['revenue_per_vehicle'].idxmin()
        ]
        
        # High probability, good outcome
        high_prob_good = self.scenario_df[
            (self.scenario_df['probability'] > 0.01) &
            (self.scenario_df['revenue_per_vehicle'] > self.risk_metrics['expected_value'])
        ].head(5)
        
        # High probability, bad outcome
        high_prob_bad = self.scenario_df[
            (self.scenario_df['probability'] > 0.01) &
            (self.scenario_df['revenue_per_vehicle'] < self.risk_metrics['expected_value'])
        ].head(5)
        
        print(f"\n📌 MOST LIKELY SCENARIO:")
        print(f"   {most_likely['scenario_id']}")
        print(f"   Probability: {most_likely['probability']*100:.2f}%")
        print(f"   Revenue: £{most_likely['revenue_per_vehicle']:.0f}/vehicle")
        
        print(f"\n🎯 BEST CASE:")
        print(f"   {best_case_scenario['scenario_id']}")
        print(f"   Revenue: £{best_case_scenario['revenue_per_vehicle']:.0f}/vehicle")
        
        print(f"\n⚠️  WORST CASE:")
        print(f"   {worst_case_scenario['scenario_id']}")
        print(f"   Revenue: £{worst_case_scenario['revenue_per_vehicle']:.0f}/vehicle")
        
        return {
            'most_likely': most_likely,
            'best_case': best_case_scenario,
            'worst_case': worst_case_scenario,
            'high_prob_good': high_prob_good,
            'high_prob_bad': high_prob_bad
        }
    
    def create_visualizations(self):
        """
        Create 2×2 risk visualization dashboard:
        - Top left: Histogram (probability-weighted distribution)
        - Top right: Cumulative Distribution Function
        - Bottom left: Tornado diagram (sensitivity)
        - Bottom right: Risk matrix (impact × probability)
        
        NO text boxes, NO waterfall charts, just clean graphs.
        """
        
        print("\n" + "="*70)
        print("📊 CREATING RISK VISUALIZATIONS")
        print("="*70)
        
        # Create 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk-Based Scenario Analysis: Revenue Distribution & Sensitivity',
                    fontsize=16, fontweight='bold', y=0.995)
        
        baseline_revenue = self.baseline_results['business_case']['revenue_per_vehicle_annual']
        
        # =================================================================
        # TOP LEFT: HISTOGRAM (Probability-Weighted Distribution)
        # =================================================================
        ax = axes[0, 0]
        
        # Create bins
        bins = np.linspace(self.scenario_df['revenue_per_vehicle'].min(),
                          self.scenario_df['revenue_per_vehicle'].max(), 30)
        
        # Weight by probability
        ax.hist(self.scenario_df['revenue_per_vehicle'], bins=bins,
               weights=self.scenario_df['probability'],
               color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Key metrics vertical lines
        ax.axvline(baseline_revenue, color='gray', linestyle='--', linewidth=2,
                  label=f'Baseline: £{baseline_revenue:.0f}', alpha=0.7)
        ax.axvline(self.risk_metrics['expected_value'], color='green', linestyle='-', linewidth=2.5,
                  label=f'Expected: £{self.risk_metrics["expected_value"]:.0f}')
        ax.axvline(self.risk_metrics['var_5_percentile'], color='red', linestyle='--', linewidth=2,
                  label=f'VaR (5%): £{self.risk_metrics["var_5_percentile"]:.0f}')
        ax.axvline(self.risk_metrics['upside_95_percentile'], color='orange', linestyle='--', linewidth=2,
                  label=f'Upside (95%): £{self.risk_metrics["upside_95_percentile"]:.0f}')
        
        ax.set_xlabel('Revenue per Vehicle (£/year)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Probability Density', fontweight='bold', fontsize=11)
        ax.set_title('Probability-Weighted Revenue Distribution', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # =================================================================
        # TOP RIGHT: CUMULATIVE DISTRIBUTION FUNCTION
        # =================================================================
        ax = axes[0, 1]
        
        # Sort by revenue
        sorted_scenarios = self.scenario_df.sort_values('revenue_per_vehicle')
        cumulative_prob = sorted_scenarios['probability'].cumsum()
        
        # Plot CDF
        ax.plot(sorted_scenarios['revenue_per_vehicle'], cumulative_prob * 100,
               color='#2ECC71', linewidth=3, alpha=0.8)
        ax.fill_between(sorted_scenarios['revenue_per_vehicle'], 0, cumulative_prob * 100,
                       alpha=0.2, color='#2ECC71')
        
        # Horizontal lines for percentiles
        ax.axhline(5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(50, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Vertical lines for key values
        ax.axvline(self.risk_metrics['var_5_percentile'], color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(self.risk_metrics['median'], color='blue', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(self.risk_metrics['upside_95_percentile'], color='orange', linestyle=':', linewidth=2, alpha=0.7)
        
        # Annotations
        ax.text(self.risk_metrics['var_5_percentile'], 5, f'  £{self.risk_metrics["var_5_percentile"]:.0f}',
               fontsize=9, va='center', color='red', fontweight='bold')
        ax.text(self.risk_metrics['median'], 50, f'  £{self.risk_metrics["median"]:.0f}',
               fontsize=9, va='center', color='blue', fontweight='bold')
        ax.text(self.risk_metrics['upside_95_percentile'], 95, f'  £{self.risk_metrics["upside_95_percentile"]:.0f}',
               fontsize=9, va='center', color='orange', fontweight='bold')
        
        ax.set_xlabel('Revenue per Vehicle (£/year)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Cumulative Probability (%)', fontweight='bold', fontsize=11)
        ax.set_title('Cumulative Distribution Function', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
        
        # =================================================================
        # BOTTOM LEFT: TORNADO DIAGRAM (Sensitivity Analysis)
        # =================================================================
        ax = axes[1, 0]
        
        # Calculate impact of each dimension
        dimensions = {}
        
        # Grid conditions
        if 'weather' in self.scenario_df.columns:
            weather_scenarios = self.scenario_df.groupby('weather')
            best_weather = weather_scenarios['revenue_per_vehicle'].mean().max()
            worst_weather = weather_scenarios['revenue_per_vehicle'].mean().min()
            dimensions['Grid Conditions\n(Event Frequency)'] = {'best': best_weather, 'worst': worst_weather}
        
        # Device performance
        if 'device' in self.scenario_df.columns:
            device_scenarios = self.scenario_df.groupby('device')
            best_device = device_scenarios['revenue_per_vehicle'].mean().max()
            worst_device = device_scenarios['revenue_per_vehicle'].mean().min()
            dimensions['Device Performance\n(Uptime)'] = {'best': best_device, 'worst': worst_device}
        
        # Market competition
        if 'competition' in self.scenario_df.columns:
            comp_scenarios = self.scenario_df.groupby('competition')
            best_comp = comp_scenarios['revenue_per_vehicle'].mean().max()
            worst_comp = comp_scenarios['revenue_per_vehicle'].mean().min()
            dimensions['Market Competition\n(Pricing)'] = {'best': best_comp, 'worst': worst_comp}
        
        # Forecasting accuracy
        if 'forecasting' in self.scenario_df.columns:
            forecast_scenarios = self.scenario_df.groupby('forecasting')
            best_forecast = forecast_scenarios['revenue_per_vehicle'].mean().max()
            worst_forecast = forecast_scenarios['revenue_per_vehicle'].mean().min()
            dimensions['Forecasting Accuracy\n(SAF Penalty)'] = {'best': best_forecast, 'worst': worst_forecast}
        
        # Sort by total range
        sorted_dims = sorted(dimensions.items(), 
                           key=lambda x: x[1]['best'] - x[1]['worst'], 
                           reverse=True)
        
        y_pos = np.arange(len(sorted_dims))
        
        for i, (name, values) in enumerate(sorted_dims):
            left = values['worst'] - baseline_revenue
            right = values['best'] - baseline_revenue
            
            # Left bar (downside)
            ax.barh(i, left, left=0, height=0.6, color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1.5)
            # Right bar (upside)
            ax.barh(i, right, left=0, height=0.6, color='#2ECC71', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Labels
            ax.text(left - 5, i, f'£{values["worst"]:.0f}', ha='right', va='center', fontsize=9, fontweight='bold')
            ax.text(right + 5, i, f'£{values["best"]:.0f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name for name, _ in sorted_dims], fontsize=10)
        ax.set_xlabel('Impact on Revenue (£/vehicle, deviation from baseline)', fontweight='bold', fontsize=10)
        ax.set_title('Tornado Diagram: Factor Sensitivity', fontweight='bold', fontsize=12)
        ax.axvline(0, color='black', linewidth=2)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # =================================================================
        # BOTTOM RIGHT: RISK MATRIX (Impact vs Probability)
        # =================================================================
        ax = axes[1, 1]
        
        # Aggregate scenarios by primary risk factor
        risk_factors = []
        
        # Grid conditions
        if 'weather' in self.scenario_df.columns:
            for scenario_type in self.scenario_df['weather'].unique():
                subset = self.scenario_df[self.scenario_df['weather'] == scenario_type]
                risk_factors.append({
                    'name': scenario_type.replace('_', ' ').title()[:10],
                    'probability': subset['probability'].sum(),
                    'impact': (subset['revenue_per_vehicle'].mean() / baseline_revenue - 1) * 100,
                    'category': 'Grid'
                })
        
        # Device performance
        if 'device' in self.scenario_df.columns:
            for scenario_type in self.scenario_df['device'].unique():
                subset = self.scenario_df[self.scenario_df['device'] == scenario_type]
                risk_factors.append({
                    'name': scenario_type.replace('_', ' ').title()[:10],
                    'probability': subset['probability'].sum(),
                    'impact': (subset['revenue_per_vehicle'].mean() / baseline_revenue - 1) * 100,
                    'category': 'Device'
                })
        
        # Market competition
        if 'competition' in self.scenario_df.columns:
            for scenario_type in self.scenario_df['competition'].unique():
                subset = self.scenario_df[self.scenario_df['competition'] == scenario_type]
                risk_factors.append({
                    'name': scenario_type.replace('_', ' ').title()[:10],
                    'probability': subset['probability'].sum(),
                    'impact': (subset['revenue_per_vehicle'].mean() / baseline_revenue - 1) * 100,
                    'category': 'Competition'
                })
        
        # Forecasting accuracy
        if 'forecasting' in self.scenario_df.columns:
            for scenario_type in self.scenario_df['forecasting'].unique():
                subset = self.scenario_df[self.scenario_df['forecasting'] == scenario_type]
                risk_factors.append({
                    'name': scenario_type.replace('_', ' ').title()[:10],
                    'probability': subset['probability'].sum(),
                    'impact': (subset['revenue_per_vehicle'].mean() / baseline_revenue - 1) * 100,
                    'category': 'Forecasting'
                })
        
        rf_df = pd.DataFrame(risk_factors)
        
        # Plot
        colors = {'Grid': '#E74C3C', 'Device': '#F39C12', 'Competition': '#3498DB', 'Forecasting': '#9B59B6'}
        
        for category in colors:
            if category in rf_df['category'].values:
                subset = rf_df[rf_df['category'] == category]
                ax.scatter(subset['probability'] * 100, subset['impact'],
                          s=200, alpha=0.7, c=colors[category], edgecolors='black', linewidth=1.5,
                          label=category)
                
                # Labels
                for _, row in subset.iterrows():
                    ax.text(row['probability'] * 100, row['impact'], 
                           row['name'], fontsize=7, ha='center', va='center')
        
        # Quadrant lines
        ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(50, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Probability (%)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Impact on Revenue (%)', fontweight='bold', fontsize=11)
        ax.set_title('Risk Matrix: Impact vs Probability', fontweight='bold', fontsize=12)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # =================================================================
        # SAVE FIGURE
        # =================================================================
        plt.tight_layout()
        
        output_path = Path('outputs')
        output_path.mkdir(exist_ok=True)
        
        fig_path = output_path / 'risk_scenario_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"   ✅ Saved visualization: {fig_path}")
        print(f"   Layout: 2×2 grid (Histogram, CDF, Tornado, Risk Matrix)")
        
        plt.close()
    
    def create_scenario_comparison_table(self):
        """
        Create detailed scenario comparison table for export.
        """
        
        print("\n" + "="*70)
        print("📋 CREATING SCENARIO COMPARISON TABLE")
        print("="*70)
        
        # Select top scenarios by probability and extreme cases
        top_10 = self.scenario_df.head(10).copy()
        
        # Add formatted columns
        baseline_revenue = self.baseline_results['business_case']['revenue_per_vehicle_annual']
        
        top_10['revenue_formatted'] = top_10['revenue_per_vehicle'].apply(lambda x: f"£{x:.0f}")
        top_10['probability_pct'] = top_10['probability'].apply(lambda x: f"{x*100:.2f}%")
        top_10['vs_baseline'] = top_10['revenue_per_vehicle'].apply(
            lambda x: f"{((x/baseline_revenue - 1)*100):+.1f}%"
        )
        
        # Select columns for export
        export_cols = [
            'scenario_id', 'probability_pct', 'revenue_formatted', 'vs_baseline',
            'weather', 'device', 'competition', 'forecasting'
        ]
        
        comparison_table = top_10[export_cols].copy()
        comparison_table.columns = [
            'Scenario', 'Probability', 'Revenue', 'vs Baseline',
            'Grid', 'Device', 'Competition', 'Forecasting'
        ]
        
        # Save
        output_path = Path('outputs')
        output_path.mkdir(exist_ok=True)
        
        table_path = output_path / 'scenario_comparison_table.csv'
        comparison_table.to_csv(table_path, index=False)
        
        print(f"   ✅ Saved comparison table: {table_path}")
        print(f"   Contains top 10 scenarios by probability")
        
        return comparison_table
    
    def generate_business_recommendations(self):
        """
        Generate actionable business recommendations based on risk analysis.
        """
        
        print("\n" + "="*70)
        print("💡 GENERATING BUSINESS RECOMMENDATIONS")
        print("="*70)
        
        recommendations = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        baseline_revenue = self.baseline_results['business_case']['revenue_per_vehicle_annual']
        
        # CRITICAL: Grid condition volatility (uncontrollable)
        recommendations['CRITICAL'].append({
            'priority': 'CRITICAL',
            'action': 'Geographic Diversification: Expand to 3+ UKPN zones',
            'rationale': f'Grid conditions drive {self.risk_metrics["range_multiplier"]:.1f}× revenue variance. Uncontrollable external risk.',
            'impact': 'Reduces event frequency volatility by 40-60%',
            'timeframe': 'Q1-Q2 2026',
            'cost': 'Medium (operational expansion)'
        })
        
        # HIGH: Device performance (controllable)
        if self.risk_metrics['std_dev'] > 30:
            recommendations['HIGH'].append({
                'priority': 'HIGH',
                'action': 'Implement 99% Device Uptime SLA with OEM Partners',
                'rationale': f'Device failures cause 33% revenue loss in critical scenarios. Controllable through partner agreements.',
                'impact': f'Prevents £{0.33 * self.risk_metrics["expected_value"]:.0f}/vehicle downside',
                'timeframe': 'Q1 2026',
                'cost': 'Low (contractual terms)'
            })
        
        # HIGH: Forecasting accuracy (controllable)
        recommendations['HIGH'].append({
            'priority': 'HIGH',
            'action': 'Invest in ML-Based Baseline Forecasting System',
            'rationale': 'Poor forecasting (85% accuracy) = 30% SAF penalty. Improving to 95% recovers full revenue.',
            'impact': f'Potential uplift: £{self.risk_metrics["expected_value"] * 0.10:.0f}/vehicle',
            'timeframe': 'Q2 2026',
            'cost': 'Medium (£50k-100k system investment)'
        })
        
        # MEDIUM: Portfolio risk buffer
        buffer_multiple = self.risk_metrics['var_5_percentile'] / baseline_revenue
        recommendations['MEDIUM'].append({
            'priority': 'MEDIUM',
            'action': f'Revenue Volatility Buffer: Maintain {buffer_multiple:.1f}× Monthly Reserves',
            'rationale': f'5th percentile = £{self.risk_metrics["var_5_percentile"]:.0f} ({buffer_multiple:.1f}× baseline). Protects against downside scenarios.',
            'impact': 'Ensures operational continuity in worst-case months',
            'timeframe': 'Immediate',
            'cost': f'Cash reserve: £{self.risk_metrics["var_5_percentile"] * self.baseline_results["business_case"]["fleet_size_participating"]:.0f} (one-time)'
        })
        
        # MEDIUM: Competition monitoring
        recommendations['MEDIUM'].append({
            'priority': 'MEDIUM',
            'action': 'Differentiate Through Service Quality vs Price Competition',
            'rationale': 'Market saturation scenarios show 15% revenue loss. Compete on reliability, not price.',
            'impact': 'Maintains margins during competitive periods',
            'timeframe': 'Ongoing',
            'cost': 'Low (operational focus)'
        })
        
        # LOW: Upside capture
        recommendations['LOW'].append({
            'priority': 'LOW',
            'action': 'Prepare Rapid Capacity Scaling for Crisis Events',
            'rationale': f'Harsh winter scenarios offer {self.risk_metrics["upside_95_percentile"]/baseline_revenue:.1f}× baseline revenue.',
            'impact': 'Captures upside during favorable conditions',
            'timeframe': 'Q3 2026',
            'cost': 'Low (contingency planning)'
        })
        
        # Print recommendations
        print("\n🎯 CRITICAL ACTIONS:")
        for category, recs in recommendations.items():
            if recs:
                print(f"\n   {category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"      [{rec['priority']}] {rec['action']}")
                    print(f"              → {rec['rationale']}")
        
        return recommendations
    
    def save_results(self, output_dir='data'):
        """
        Save all scenario results to files.
        """
        
        print("\n" + "="*70)
        print("💾 SAVING RESULTS")
        print("="*70)
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Save full scenario results
        output_path = Path(output_dir) / 'risk_scenario_results.csv'
        self.scenario_df.to_csv(output_path, index=False)
        print(f"   ✅ Saved full scenario results: {output_path}")
        
        # 2. Save risk metrics
        metrics_path = Path(output_dir) / 'risk_metrics_summary.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.risk_metrics, f, indent=2)
        print(f"   ✅ Saved risk metrics: {metrics_path}")
        
        # 3. Save risk dimension details
        risk_dims = {
            'weather_scenarios': self.weather_scenarios,
            'device_scenarios': self.device_scenarios,
            'competition_scenarios': self.competition_scenarios,
            'forecasting_scenarios': self.forecasting_scenarios
        }
        dims_path = Path(output_dir) / 'risk_dimensions.json'
        with open(dims_path, 'w') as f:
            json.dump(risk_dims, f, indent=2)
        print(f"   ✅ Saved risk dimensions: {dims_path}")
    
    def run_complete_analysis(self):
        """
        Execute complete risk-based scenario analysis.
        """
        
        print("\n" + "="*70)
        print("🚀 RUNNING COMPLETE RISK-BASED SCENARIO ANALYSIS")
        print("="*70)
        
        # Step 1: Define scenarios
        self.define_risk_scenarios()
        
        # Step 2: Calculate combinations
        self.calculate_combined_scenarios()
        
        # Step 3: Calculate expected value
        self.calculate_expected_value()
        
        # Step 4: Identify key scenarios
        key_scenarios = self.identify_key_scenarios()
        
        # Step 5: Generate visualizations
        self.create_visualizations()
        
        # Step 6: Create comparison table
        self.create_scenario_comparison_table()
        
        # Step 7: Generate recommendations
        recommendations = self.generate_business_recommendations()
        
        # Step 8: Save all results
        self.save_results()
        
        print("\n" + "="*70)
        print("✅ RISK-BASED SCENARIO ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n📊 Key Outputs:")
        print(f"   - Risk scenario analysis chart (PNG) - 2×2 layout")
        print(f"   - Scenario comparison table (CSV)")
        print(f"   - Risk metrics summary (JSON)")
        print(f"   - Full scenario results (CSV)")
        
        return {
            'scenario_results': self.scenario_df,
            'risk_metrics': self.risk_metrics,
            'key_scenarios': key_scenarios,
            'recommendations': recommendations
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for risk-based scenario analysis.
    """
    
    print("\n" + "="*70)
    print("🎯 EV FLEXIBILITY BIDDING: RISK-BASED SCENARIO ANALYSIS")
    print("="*70)
    print("\nThis analysis quantifies business risks across 4 dimensions:")
    print("  1. Grid Conditions (Event Frequency) - Uncontrollable")
    print("  2. Device Performance (Uptime) - Partially Controllable")
    print("  3. Market Competition (Pricing) - Uncontrollable")
    print("  4. Forecasting Accuracy (SAF Penalties) - Controllable")
    print("\nOutputs:")
    print("  - Probability-weighted expected value")
    print("  - Risk exposure metrics (VaR, percentiles)")
    print("  - 2×2 visualization (Histogram, CDF, Tornado, Risk Matrix)")
    print("  - Actionable business recommendations")
    print("="*70)
    
    # Initialize engine
    engine = RiskScenarioAnalysis(
        baseline_results_path='data/business_case_summary.json'
    )
    
    # Run complete analysis
    results = engine.run_complete_analysis()
    
    # Print final summary
    print("\n" + "="*70)
    print("📈 FINAL SUMMARY FOR PORTFOLIO")
    print("="*70)
    
    baseline_revenue = engine.baseline_results['business_case']['revenue_per_vehicle_annual']
    
    print(f"\n💰 REVENUE ANALYSIS:")
    print(f"   Baseline (deterministic model): £{baseline_revenue:.0f}/vehicle")
    print(f"   Expected Value (risk-adjusted): £{results['risk_metrics']['expected_value']:.0f}/vehicle")
    print(f"   Difference: {results['risk_metrics']['ev_vs_baseline_pct']:+.1f}%")
    
    print(f"\n📊 RISK EXPOSURE:")
    print(f"   Worst Case: £{results['risk_metrics']['worst_case']:.0f}/vehicle (-{(1-results['risk_metrics']['worst_case']/baseline_revenue)*100:.0f}%)")
    print(f"   5th Percentile: £{results['risk_metrics']['var_5_percentile']:.0f}/vehicle (-{(1-results['risk_metrics']['var_5_percentile']/baseline_revenue)*100:.0f}%)")
    print(f"   95th Percentile: £{results['risk_metrics']['upside_95_percentile']:.0f}/vehicle (+{(results['risk_metrics']['upside_95_percentile']/baseline_revenue-1)*100:.0f}%)")
    print(f"   Best Case: £{results['risk_metrics']['best_case']:.0f}/vehicle (+{(results['risk_metrics']['best_case']/baseline_revenue-1)*100:.0f}%)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   - Grid conditions are PRIMARY risk driver ({results['risk_metrics']['range_multiplier']:.1f}× variance)")
    print(f"   - Device uptime and forecasting are CONTROLLABLE (99% SLA + ML forecasting)")
    print(f"   - Portfolio risk exposure: {results['risk_metrics']['var_5_percentile']/baseline_revenue:.1f}× baseline volatility")
    print(f"   - Upside potential: {results['risk_metrics']['upside_95_percentile']/baseline_revenue:.1f}× baseline")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE - READY FOR PORTFOLIO!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()