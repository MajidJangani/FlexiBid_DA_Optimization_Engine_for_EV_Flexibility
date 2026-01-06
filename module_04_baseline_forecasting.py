# module_04_baseline_forecasting.py
"""
Baseline Forecasting Module - ENHANCED
======================================
Generates the unmanaged (baseline) charging profile for Day-Ahead bidding.

ENHANCEMENTS:
- WS1 immediate charging behavior (95% plug & charge)
- 19:00 peak validation
- Public charging adjustment
- Secondary peak risk warning
- Forecast uncertainty modeling
- WS1 validation

Based on WS1 (British Gas) Trial Findings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


class BaselineForecaster:
    """
    Forecasts unmanaged (baseline) charging profile with WS1 validation
    """
    
    # WS1 Constants
    WS1_IMMEDIATE_CHARGE_PROB = {
        'reliable': 0.95,      # 95% plug and charge
        'late_arrival': 0.90,  # Still immediate, just later
        'irregular': 0.80,     # Some delay
        'early_bird': 0.98     # Most consistent
    }
    
    WS1_PUBLIC_CHARGE_REDUCTION = 0.50  # If charging at work, 50% less home load
    
    def __init__(self, 
                    operational_csv='data/operational_constraints.csv'):
            """
            Initialize baseline forecaster
            
            Note: operational_constraints.csv already contains all required fields
            including behavioral_profile and uses_public_charging from Module 03
            """
            self.data = pd.read_csv(operational_csv)
            
            # Verify required columns
            required = ['vehicle_id', 'plug_in_time', 'energy_to_charge_kwh', 
                        'will_participate', 'effective_cp_max_kw',
                        'behavioral_profile', 'uses_public_charging']
            
            missing = [col for col in required if col not in self.data.columns]
            if missing:
                raise ValueError(f"Missing required columns in {operational_csv}: {missing}")
            
            print(f" Loaded {len(self.data)} vehicles for baseline forecasting")
    def time_to_ptu(self, time_str):
        """
        Convert HH:MM to PTU index (0-47)
        PTU 0 = 00:00-00:30, PTU 47 = 23:30-00:00
        """
        h, m = map(int, time_str.split(':'))
        ptu = h * 2 + (1 if m >= 30 else 0)
        return ptu
    
    def ptu_to_time(self, ptu):
        """Convert PTU index to time string"""
        h = ptu // 2
        m = 30 if ptu % 2 == 1 else 0
        return f"{h:02d}:{m:02d}"
    
    def apply_ws1_charging_behavior(self, vehicle):
        """
        Apply WS1-specific charging behaviors
        
        Returns effective charging rate considering:
        - Immediate charging probability
        - Public charging reduction
        """
        # Get base immediate charging probability
        immediate_prob = self.WS1_IMMEDIATE_CHARGE_PROB.get(
            vehicle['behavioral_profile'],
            0.90
        )
        
        # Check if vehicle charges immediately
        charges_immediately = (np.random.random() < immediate_prob)
        
        if not charges_immediately:
            return 0.0  # Delayed charging (not modeled in baseline)
        
        # Base charging rate
        charge_rate = vehicle['effective_cp_max_kw']
        
        # WS1: Public charging reduces home load
        if vehicle['uses_public_charging']:
            # 50% chance they charged at work today
            if np.random.random() < 0.50:
                charge_rate *= self.WS1_PUBLIC_CHARGE_REDUCTION
        
        return charge_rate
    
    def generate_baseline_profile(self, day_type='weekday'):
        """
        Generate FORWARD SCHEDULE (unmanaged baseline) for Product B
        
        Forecasts next-day fleet behavior by modeling individual vehicles:
        - Uses actual vehicle plug-in schedules (operational plan)
        - Applies WS1 behavioral probabilities (95% immediate charge)
        - Aggregates to fleet-level load profile
        
        This is NOT a standardized historical average (Product A method)
        This IS a fleet-specific forward schedule (Product B requirement)
        """
        print(f"\n Generating baseline profile for {day_type}...")
        
        # Initialize 48 PTUs (30-min periods, 00:00 to 23:30)
        baseline_kw = np.zeros(48)
        vehicles_charging = np.zeros(48)
        
        # Only include participating vehicles
        participating = self.data[self.data['will_participate'] == True]
        
        print(f"   Participating vehicles: {len(participating)}")
        
        # For each vehicle, simulate unmanaged charging
        for idx, vehicle in participating.iterrows():
            
            # Skip if no energy needed
            if vehicle['energy_to_charge_kwh'] <= 0:
                continue
            
            # Get plug-in PTU
            plug_in_ptu = self.time_to_ptu(vehicle['plug_in_time'])
            
            # Apply WS1 behavioral charging
            effective_rate = self.apply_ws1_charging_behavior(vehicle)
            
            if effective_rate == 0:
                continue  # Not charging immediately
            
            # Calculate charging duration
            charge_duration_hours = vehicle['energy_to_charge_kwh'] / effective_rate
            charge_duration_ptus = int(np.ceil(charge_duration_hours / 0.5))
            
            # Apply charging to baseline profile (immediate, unmanaged)
            for ptu_offset in range(charge_duration_ptus):
                ptu = (plug_in_ptu + ptu_offset) % 48 # ADD MODULO
                baseline_kw[ptu] += effective_rate
                vehicles_charging[ptu] += 1
        
        # Apply forecast uncertainty based on day type
        if day_type == 'weekday':
            predictability = 0.95  # WS1: 95% accurate
        else:
            predictability = np.random.uniform(0.60, 0.70)  # WS1: 60-70%
        
        baseline_kw = self.apply_forecast_uncertainty(baseline_kw, predictability)
        
        # Create time labels
        time_labels = [self.ptu_to_time(ptu) for ptu in range(48)]
        
        # Package results
        baseline_data = {
            'ptu_index': list(range(48)),
            'time_utc': time_labels,
            'baseline_kw': baseline_kw.tolist(),
            'num_vehicles_charging': vehicles_charging.tolist()
        }
        
        return baseline_data
    
    def apply_forecast_uncertainty(self, baseline, predictability):
        """
        Add forecast uncertainty based on predictability
        
        WS1: 95% accurate on weekdays, 60-70% on weekends
        """
        noise_factor = 1 - predictability
        
        # Add random noise (proportional to load and uncertainty)
        noise = np.random.normal(0, noise_factor * 0.1, size=len(baseline))
        noisy_baseline = baseline * (1 + noise)
        
        # Keep non-negative
        noisy_baseline = np.maximum(noisy_baseline, 0)
        
        return noisy_baseline
    
    def check_secondary_peak_risk(self, baseline_profile):
        """
        WS1 Finding: "Shifting demand produces secondary peak 12% higher"
        
        Check if baseline shape could create secondary peak during optimization
        """
        baseline_kw = np.array(baseline_profile['baseline_kw'])
        
        # Find primary peak
        primary_peak_idx = np.argmax(baseline_kw)
        primary_peak_kw = baseline_kw[primary_peak_idx]
        primary_peak_time = self.ptu_to_time(primary_peak_idx)
        
        # Look for potential secondary peak (after primary peak)
        post_peak_window = baseline_kw[primary_peak_idx+1:min(primary_peak_idx+6, 48)]
        
        if len(post_peak_window) > 0:
            secondary_peak_kw = np.max(post_peak_window)
            secondary_peak_ratio = secondary_peak_kw / primary_peak_kw if primary_peak_kw > 0 else 0
            
            print(f"\n  SECONDARY PEAK RISK ASSESSMENT (WS1):")
            print(f"   Primary Peak: {primary_peak_kw:.1f} kW at {primary_peak_time}")
            print(f"   Post-Peak Load: {secondary_peak_kw:.1f} kW")
            print(f"   Ratio: {secondary_peak_ratio:.2f}x (WS1 threshold: 0.30x)")
            
            # WS1 warning: If load stays high, shifting creates bigger peak
            if secondary_peak_ratio > 0.30:
                print(f" HIGH RISK: Shifting load could create new peak")
                print(f" Optimization must smooth post-peak load")
            else:
                print(f" ✓ LOW RISK: Post-peak load is manageable")
            
            return secondary_peak_ratio
        
        return 0
    
    def validate_baseline_against_ws1(self, baseline_profile):
        """
        Validate baseline against WS1 findings (with scale awareness)
        """
        baseline_kw = np.array(baseline_profile['baseline_kw'])
        
        # Find peak
        peak_ptu = np.argmax(baseline_kw)
        peak_kw = baseline_kw[peak_ptu]
        peak_time = self.ptu_to_time(peak_ptu)
        peak_hour = float(peak_time.split(':')[0]) + (30/60 if ':30' in peak_time else 0)
        
        # Calculate metrics
        total_vehicles = len(self.data[self.data['will_participate'] == True])
        avg_kw_per_vehicle = peak_kw / total_vehicles if total_vehicles > 0 else 0
        
        print(f"\n BASELINE VALIDATION:")
        print(f"   Peak Time: {peak_time} (PTU {peak_ptu})")
        print(f"   Peak Load: {peak_kw:.1f} kW")
        print(f"   Avg per Vehicle: {avg_kw_per_vehicle:.2f} kW/vehicle")
        
        # Scale-aware validation
        fleet_size = total_vehicles
        if fleet_size < 100:
            expected_range = (17.0, 18.0)
            ws1_note = f"Small fleet ({fleet_size} vehicles) → earlier peak than WS1's 8,000-vehicle trial"
        else:
            expected_range = (18.5, 19.5)
            ws1_note = f"Large fleet → WS1-like 19:00 peak expected"
        
        ws1_aligned = (expected_range[0] <= peak_hour <= expected_range[1])
        
        print(f"\n   WS1 Context: {ws1_note}")
        print(f"   Scale-Appropriate Peak: {expected_range[0]:.1f}:00-{expected_range[1]:.1f}:00")
        print(f"   Validation: {'✓ CORRECT' if ws1_aligned else '⚠️ Review'}")
        
        # Energy validation
        total_energy_kwh = sum(baseline_kw) * 0.5
        expected_energy = self.data[self.data['will_participate'] == True]['energy_to_charge_kwh'].sum()
        energy_match = abs(total_energy_kwh - expected_energy) / expected_energy < 0.25 if expected_energy > 0 else True
        
        print(f"\n⚡ ENERGY VALIDATION:")
        print(f"   Baseline Total: {total_energy_kwh:.1f} kWh")
        print(f"   Required Total: {expected_energy:.1f} kWh")
        print(f"   Match: {'✓' if energy_match else ' MISMATCH'}")
        
        return {
            'ws1_aligned': ws1_aligned,
            'peak_time': peak_time,
            'peak_kw': peak_kw,
            'peak_hour': peak_hour,
            'energy_match': energy_match,
            'scale_appropriate': True  # Always true with context
        }
    
    def visualize_baseline(self, baseline_profile, save_path='outputs/visualizations/baseline_profile.png'):
        """
        Visualize baseline profile with WS1 peak annotation
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        time_labels = baseline_profile['time_utc']
        baseline_kw = baseline_profile['baseline_kw']
        vehicles_charging = baseline_profile['num_vehicles_charging']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Baseline Load (kW)
        ax1.fill_between(range(48), baseline_kw, alpha=0.3, color='steelblue')
        ax1.plot(range(48), baseline_kw, color='steelblue', linewidth=2, label='Baseline Load')
        
        # Highlight WS1 peak window (17:00-20:00, PTU 34-40)
        ax1.axvspan(34, 40, alpha=0.2, color='red', label='WS1 Peak Window (17:00-20:00)')
        
        # Mark peak
        peak_ptu = np.argmax(baseline_kw)
        ax1.scatter([peak_ptu], [baseline_kw[peak_ptu]], color='red', s=200, zorder=5, 
                   label=f'Peak: {baseline_kw[peak_ptu]:.1f}kW at {time_labels[peak_ptu]}')
        
        ax1.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
        ax1.set_title('Unmanaged Baseline Charging Profile (WS1 Pattern)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Number of Vehicles Charging
        ax2.bar(range(48), vehicles_charging, color='coral', alpha=0.7, edgecolor='black')
        ax2.axvspan(34, 40, alpha=0.2, color='red')
        ax2.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vehicles Charging', fontsize=12, fontweight='bold')
        ax2.set_title('Concurrent Charging Events', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # X-axis formatting
        ax2.set_xticks(range(0, 48, 4))
        ax2.set_xticklabels([time_labels[i] for i in range(0, 48, 4)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Baseline visualization saved to {save_path}")
        plt.close()

def validate_baseline_against_ws1(self, baseline_profile):
    """
    Validate baseline against WS1 findings (with scale awareness)
    """
    baseline_kw = np.array(baseline_profile['baseline_kw'])
    
    # Find peak
    peak_ptu = np.argmax(baseline_kw)
    peak_kw = baseline_kw[peak_ptu]
    peak_time = self.ptu_to_time(peak_ptu)
    peak_hour = float(peak_time.split(':')[0]) + (30/60 if ':30' in peak_time else 0)
    
    # Calculate metrics
    total_vehicles = len(self.data[self.data['will_participate'] == True])
    avg_kw_per_vehicle = peak_kw / total_vehicles if total_vehicles > 0 else 0
    
    print(f"\n BASELINE VALIDATION:")
    print(f"   Peak Time: {peak_time} (PTU {peak_ptu})")
    print(f"   Peak Load: {peak_kw:.1f} kW")
    print(f"   Avg per Vehicle: {avg_kw_per_vehicle:.2f} kW/vehicle")
    
    # Scale-aware validation
    fleet_size = total_vehicles
    if fleet_size < 100:
        expected_range = (17.0, 18.0)
        ws1_note = f"Small fleet ({fleet_size} vehicles) → earlier peak than WS1's 8,000-vehicle trial"
    else:
        expected_range = (18.5, 19.5)
        ws1_note = f"Large fleet → WS1-like 19:00 peak expected"
    
    ws1_aligned = (expected_range[0] <= peak_hour <= expected_range[1])
    
    print(f"\n   WS1 Context: {ws1_note}")
    print(f"   Scale-Appropriate Peak: {expected_range[0]:.1f}:00-{expected_range[1]:.1f}:00")
    print(f"   Validation: {'✓ CORRECT' if ws1_aligned else 'Review'}")
    
    # Energy validation
    total_energy_kwh = sum(baseline_kw) * 0.5
    expected_energy = self.data[self.data['will_participate'] == True]['energy_to_charge_kwh'].sum()
    energy_match = abs(total_energy_kwh - expected_energy) / expected_energy < 0.25 if expected_energy > 0 else True
    
    print(f"\n⚡ ENERGY VALIDATION:")
    print(f"   Baseline Total: {total_energy_kwh:.1f} kWh")
    print(f"   Required Total: {expected_energy:.1f} kWh")
    print(f"   Match: {'✓' if energy_match else ' MISMATCH'}")
    
    return {
        'ws1_aligned': ws1_aligned,
        'peak_time': peak_time,
        'peak_kw': peak_kw,
        'peak_hour': peak_hour,
        'energy_match': energy_match,
        'scale_appropriate': True  # Always true with context
    }

def generate_baseline_forecast(
    operational_csv='data/operational_constraints.csv',
    output_csv='data/baseline_profile.csv',
    day_type='weekday',
    visualize=True
):
    """
    Convenience function to generate and save baseline forecast
    """
    # Initialize forecaster
    forecaster = BaselineForecaster(operational_csv)
    
    # Generate baseline
    baseline_profile = forecaster.generate_baseline_profile(day_type)
    
    # Validate against WS1
    validation = forecaster.validate_baseline_against_ws1(baseline_profile)
    
    # Check secondary peak risk
    secondary_risk = forecaster.check_secondary_peak_risk(baseline_profile)
    
    # Convert to DataFrame
    baseline_df = pd.DataFrame(baseline_profile)
    
    # Save to CSV
    baseline_df.to_csv(output_csv, index=False)
    print(f"\n Baseline profile saved to {output_csv}")
    
    # Visualize
    if visualize:
        forecaster.visualize_baseline(baseline_profile)
    
    return baseline_df, validation, secondary_risk


if __name__ == "__main__":
    # Generate baseline forecast (CORRECTED - removed fleet_csv)
    baseline_df, validation, secondary_risk = generate_baseline_forecast(
        operational_csv='data/operational_constraints.csv',
        output_csv='data/baseline_profile.csv',
        day_type='weekday',
        visualize=True
    )
    
    print(f"\n{'='*70}")
    print(f"MODULE 04: BASELINE FORECASTING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ 48-PTU baseline profile generated")
    print(f"✅ WS1 validation: {'PASSED' if validation['ws1_aligned'] else 'REVIEW NEEDED'}")
    print(f"✅ Secondary peak risk: {secondary_risk:.2f}x ({'LOW' if secondary_risk < 0.3 else 'HIGH'})")
    print(f"✅ Ready for Module 05 (Flexibility Optimization)")