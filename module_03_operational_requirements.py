# module_03_operational_requirements_FIXED.py
"""
Operational Requirements Module - 
===========================================
Calculates vehicle-specific constraints for flexibility optimization.

 SOURCE-VALIDATED 5-STEP METHODOLOGY for energy calculation
- STEP 1: Calculate operational need (travel energy)
- STEP 2: Add safety buffer (behavioral uncertainty: 10%)
- STEP 3: Calculate target SoC from energy requirements
- STEP 4: Apply bounds (30% min, 90% max for safety thresholds)
- STEP 5: Calculate energy_to_charge from SoC gap (NOT from energy subtraction)

Based on: OP Deliverable D7, MILP Constraint (Equation 7)
Reference: "energy_to_charge = (target_soc - return_soc) × battery_capacity"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# REAL UKPN CONSTRAINT ZONES (From Module 00 Market Analysis)
UKPN_CONSTRAINT_ZONES = {
    # High-activity zones (>200 events) - 50% of fleet
    'West Letchworth Shefford Biggleswade': {
        'events': 333, 'revenue': 38506, 'avg_price': 579, 
        'region': 'Eastern', 'priority': 'High'
    },
    'Trowse Grid 33': {
        'events': 325, 'revenue': 19889, 'avg_price': 236,
        'region': 'Eastern', 'priority': 'High'
    },
    'Worthing Grid A': {
        'events': 276, 'revenue': 48742, 'avg_price': 732,
        'region': 'South East', 'priority': 'High'
    },
    'Uplands Park': {
        'events': 255, 'revenue': None,
        'region': 'London', 'priority': 'High'
    },
    'Cockfosters': {
        'events': 234, 'revenue': None,
        'region': 'London', 'priority': 'High'
    },
    
    # Medium-activity zones (100-200 events) - 30% of fleet
    'Central Harpenden': {
        'events': 153, 'revenue': 12263, 'avg_price': 596,
        'region': 'Eastern', 'priority': 'Medium'
    },
    'Sundon Little Barford': {
        'events': 147, 'revenue': 13460, 'avg_price': 354,
        'region': 'Eastern', 'priority': 'Medium'
    },
    'Capel Switching Station': {
        'events': 132, 'revenue': None,
        'region': 'South East', 'priority': 'Medium'
    },
    'Brandon': {
        'events': 118, 'revenue': None,
        'region': 'Eastern', 'priority': 'Medium'
    },
    'Bramford Diss Thetford': {
        'events': 112, 'revenue': None,
        'region': 'Eastern', 'priority': 'Medium'
    },
    
    # Low-activity zones (<100 events) - 20% of fleet
    'Chartham': {'events': 98, 'region': 'South East', 'priority': 'Low'},
    'Manton Lane': {'events': 91, 'region': 'Eastern', 'priority': 'Low'},
    'Haverhill': {'events': 82, 'region': 'Eastern', 'priority': 'Low'},
    'Canvey': {'events': 74, 'region': 'South East', 'priority': 'Low'},
    'Buckingham Road': {'events': 73, 'region': 'London', 'priority': 'Low'},
    'Hendon Way': {'events': 71, 'region': 'London', 'priority': 'Low'},
    'Brockenhurst Mill Hill Total': {'events': 70, 'region': 'London', 'priority': 'Low'},
    'Wrotham': {'events': 58, 'region': 'South East', 'priority': 'Low'},
    'Thorpe Grid 33': {'events': 54, 'region': 'Eastern', 'priority': 'Low'},
    'March Primary': {'events': 30, 'region': 'Eastern', 'priority': 'Low'}
}


class OperationalRequirementsCalculator:
    """
    Calculates operational constraints for EV fleet with UKPN zone mapping
    """
    
    # WS1 Constants
    WS1_ABSOLUTE_MIN_SOC = 0.20  # 20% absolute minimum
    WS1_RANGE_ANXIETY_BUFFER = 0.10  # +10% buffer for range anxiety
    WS1_MAX_SOC_TARGET = 0.90  # 90% maximum (preserves battery life)
    WS1_OPERATIONAL_BUFFER = 0.10  # 10% operational buffer (WS1-aligned, applied to travel energy)
    WS1_BEHAVIORAL_BUFFER = 0.05  # 5% behavioral buffer (applied to charging energy)
    WS1_PUBLIC_CHARGE_REDUCTION = (0.3, 0.7)  # 30-70% of daily needs
    
    def __init__(self, fleet_csv='data/synthetic_fleet.csv'):
        """Initialize calculator with fleet data"""
        self.fleet_df = pd.read_csv(fleet_csv)
        print(f" Loaded fleet of {len(self.fleet_df)} vehicles")
        
        
    def assign_ukpn_zones(self, fleet_df):
        """
        Assign realistic UKPN constraint zones to fleet
        Weighting based on market activity:
        - High zones (>200 events): 50% of fleet
        - Medium zones (100-200): 30% of fleet
        - Low zones (<100): 20% of fleet
        """
        print(f"\n  Assigning UKPN constraint zones...")
        
        # Categorize zones by activity
        high_zones = [z for z, d in UKPN_CONSTRAINT_ZONES.items() if d['events'] > 200]
        medium_zones = [z for z, d in UKPN_CONSTRAINT_ZONES.items() if 100 <= d['events'] <= 200]
        low_zones = [z for z, d in UKPN_CONSTRAINT_ZONES.items() if d['events'] < 100]
        
        num_vehicles = len(fleet_df)
        
        # Calculate allocations
        high_count = int(num_vehicles * 0.50)
        medium_count = int(num_vehicles * 0.30)
        low_count = num_vehicles - high_count - medium_count
        
        # Assign zones with weights
        zone_assignments = []
        
        # High-activity zones (more vehicles)
        zone_assignments.extend(
            np.random.choice(high_zones, size=high_count, replace=True)
        )
        
        # Medium-activity zones
        zone_assignments.extend(
            np.random.choice(medium_zones, size=medium_count, replace=True)
        )
        
        # Low-activity zones
        zone_assignments.extend(
            np.random.choice(low_zones, size=low_count, replace=True)
        )
        
        # Shuffle and assign
        np.random.shuffle(zone_assignments)
        fleet_df['ukpn_constraint_zone'] = zone_assignments
        
        # Add zone metadata
        fleet_df['zone_region'] = fleet_df['ukpn_constraint_zone'].map(
            lambda z: UKPN_CONSTRAINT_ZONES[z]['region']
        )
        fleet_df['zone_priority'] = fleet_df['ukpn_constraint_zone'].map(
            lambda z: UKPN_CONSTRAINT_ZONES[z]['priority']
        )
        
        print(f"    Assigned {len(zone_assignments)} vehicles to {fleet_df['ukpn_constraint_zone'].nunique()} zones")
        print(f"\n   Zone Distribution:")
        zone_counts = fleet_df['ukpn_constraint_zone'].value_counts()
        for zone, count in zone_counts.head(5).items():
            print(f"      {zone[:40]}: {count} vehicles")
        if len(zone_counts) > 5:
            print(f"      ... and {len(zone_counts) - 5} more zones")
        
        return fleet_df
    
    def time_to_minutes(self, time_str):
        """Convert HH:MM to minutes since midnight"""
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    
    def minutes_to_time(self, minutes):
        """Convert minutes since midnight to HH:MM"""
        h = int(minutes // 60) % 24
        m = int(minutes % 60)
        return f"{h:02d}:{m:02d}"
    
    def calculate_required_energy(self, vehicle):
        """
        Calculate energy required using SOURCE-VALIDATED 5-STEP METHODOLOGY
        
        Based on: OP Deliverable D7, MILP Constraint (Equation 7)
        
        STEP 1: Calculate operational need (travel energy)
        STEP 2: Add safety buffer (behavioral uncertainty)
        STEP 3: Calculate target SoC
        STEP 4: Apply bounds (safety thresholds)
        STEP 5: Calculate energy to charge from SoC gap
        """
        # ============================================================
        # STEP 1: CALCULATE OPERATIONAL NEED
        # ============================================================
        # Base travel energy
        travel_energy_kwh = (
            vehicle['daily_mileage_km'] * 
            vehicle['base_efficiency_wh_km'] * 
            vehicle['seasonal_efficiency_factor'] / 1000
        )
        
        # WS1: 23.3% use public charging (reduces home charging need)
        public_charge_kwh = 0
        if vehicle.get('uses_public_charging', False):
            # 50% chance they charged at work today
            if np.random.random() < 0.50:
                public_charge_ratio = np.random.uniform(*self.WS1_PUBLIC_CHARGE_REDUCTION)
                public_charge_kwh = travel_energy_kwh * public_charge_ratio
        
        # Home energy requirement (what needs to be replaced)
        home_energy_kwh = travel_energy_kwh - public_charge_kwh
        
        # ============================================================
        # STEP 2: ADD SAFETY BUFFER (Behavioral Uncertainty)
        # ============================================================
        # STEP 2: ADD OPERATIONAL BUFFER (10%, WS1-ALIGNED)
        # Applied to travel energy to account for:
        # - Route deviations (longer actual distance)
        # - Traffic delays (stop-go inefficiency)
        # - Forecasting error in mileage prediction
        required_energy_kwh = home_energy_kwh * (1 + self.WS1_OPERATIONAL_BUFFER)
        
        # ============================================================
        # STEP 3: CALCULATE TARGET SOC
        # ============================================================
        # Current energy in battery
        current_energy_kwh = vehicle['return_soc'] * vehicle['usable_battery_kwh']
        
        # Target energy = current + what we need to add
        target_energy_kwh = current_energy_kwh + required_energy_kwh
        
        # Convert to SoC
        target_soc = target_energy_kwh / vehicle['usable_battery_kwh']
        
        # ============================================================
        # STEP 4: APPLY BOUNDS (Safety Thresholds)
        # ============================================================
        # Minimum: 20% absolute + 10% range anxiety buffer = 30%
        MIN_SOC = self.WS1_ABSOLUTE_MIN_SOC + self.WS1_RANGE_ANXIETY_BUFFER  # 0.30
        
        # Maximum: 90% (sources say 95%, but 90% preserves battery life)
        MAX_SOC = self.WS1_MAX_SOC_TARGET  # 0.90
        
        target_soc = np.clip(target_soc, MIN_SOC, MAX_SOC)
        
        # ============================================================
        # STEP 5: CALCULATE ENERGY TO CHARGE (SoC Gap Method)
        # ============================================================
        # STEP 5: CALCULATE ENERGY TO CHARGE (SoC Gap Method + Behavioral Buffer)
        # SOURCE FORMULA: energy_to_charge = (SOCd - SOCa) × battery_capacity
        soc_gap = target_soc - vehicle['return_soc']

        # Apply behavioral buffer (5%) to account for:
        # - Driver forgets to plug in immediately
        # - Unexpected additional trips
        # - CP reliability issues (6A minimum stability)
        # - Battery degradation in mixed-age fleet
        energy_to_charge_kwh = max(
            soc_gap * vehicle['usable_battery_kwh'] * (1 + self.WS1_BEHAVIORAL_BUFFER), 
            0
        )

        # TOTAL BUFFER: 1.10 (operational) × 1.05 (behavioral) = 1.155 (15.5%)
        
        return {
            'travel_energy_kwh': round(travel_energy_kwh, 2),
            'public_charge_kwh': round(public_charge_kwh, 2),
            'home_energy_kwh': round(home_energy_kwh, 2),
            'required_energy_kwh': round(required_energy_kwh, 2),
            'current_energy_kwh': round(current_energy_kwh, 2),
            'target_soc': round(target_soc, 3),
            'energy_to_charge_kwh': round(energy_to_charge_kwh, 2)
        }
    
    def calculate_time_constraints(self, vehicle, energy_to_charge_kwh):
        """Calculate temporal constraints"""
        plug_in_min = self.time_to_minutes(vehicle['plug_in_time'])
        plug_out_min = self.time_to_minutes(vehicle['plug_out_time'])
        
        if plug_out_min <= plug_in_min:
            plug_out_min += 1440
        
        available_charge_time_min = plug_out_min - plug_in_min
        available_charge_time_hours = available_charge_time_min / 60
        
        effective_cp_max = vehicle['effective_cp_max_kw']
        
        if effective_cp_max > 0 and energy_to_charge_kwh > 0:
            minimum_charge_time_hours = energy_to_charge_kwh / effective_cp_max
        else:
            minimum_charge_time_hours = 0
        
        flexibility_margin_hours = available_charge_time_hours - minimum_charge_time_hours
        
        critical_latest_start_min = plug_out_min - int(minimum_charge_time_hours * 60)
        critical_latest_start = self.minutes_to_time(critical_latest_start_min % 1440)
        
        is_operational_feasible = (flexibility_margin_hours >= 0)
        
        return {
            'available_charge_time_hours': round(available_charge_time_hours, 2),
            'minimum_charge_time_hours': round(minimum_charge_time_hours, 2),
            'flexibility_margin_hours': round(flexibility_margin_hours, 2),
            'critical_latest_start': critical_latest_start,
            'is_operational_feasible': is_operational_feasible
        }
    
    def check_cp_constraints(self, vehicle, energy_to_charge_kwh, available_hours):
        """
        Check CP minimum power constraint (1.4kW stability limit)
        
        CRITICAL: CP cannot charge below 1.4kW or it "hunts" (on/off cycles)
        """
        if available_hours > 0 and energy_to_charge_kwh > 0:
            required_avg_kw = energy_to_charge_kwh / available_hours
        else:
            required_avg_kw = 0
        
        # CP constraint: Either 0 kW (off) or >= 1.4 kW
        cp_constraint_violation = (0 < required_avg_kw < vehicle['cp_min_kw'])
        
        return {
            'required_avg_kw': round(required_avg_kw, 2),
            'cp_constraint_violation': cp_constraint_violation,
            'cp_min_kw': vehicle['cp_min_kw']
        }
    
    def apply_opt_out_logic(self, vehicle):
        """Apply behavioral opt-out probabilities"""
        will_participate = np.random.random() >= vehicle['opt_out_probability']
        
        opt_out_reason = None
        if not will_participate:
            if vehicle['behavioral_profile'] == 'irregular':
                opt_out_reason = 'driver_unreliable'
            elif vehicle['behavioral_profile'] == 'late_arrival':
                opt_out_reason = 'late_arrival_risk'
            else:
                opt_out_reason = 'driver_preference'
        
        return will_participate, opt_out_reason
    
    def calculate_flexibility_score(self, vehicle_record):
        """
        Calculate vehicle flexibility readiness score (0-100)
        
        Based on:
        - Flexibility margin (40 points)
        - Driver reliability (30 points)
        - Energy buffer (20 points)
        - CP capacity (10 points)
        """
        score = 0
        
        # 1. Flexibility margin (0-40 points)
        margin_hours = vehicle_record.get('flexibility_margin_hours', 0)
        score += min(margin_hours / 10 * 40, 40)
        
        # 2. Reliability (0-30 points)
        reliability = vehicle_record.get('driver_reliability', 0.5)
        score += reliability * 30
        
        # 3. Energy buffer (0-20 points)
        if vehicle_record['required_energy_kwh'] > 0:
            buffer_ratio = ((vehicle_record['required_energy_kwh'] - 
                           vehicle_record['travel_energy_kwh']) / 
                           vehicle_record['travel_energy_kwh'])
            score += min(buffer_ratio * 100, 20)
        
        # 4. CP capacity (0-10 points)
        cp_kw = vehicle_record.get('effective_cp_max_kw', 0)
        score += min(cp_kw / 11 * 10, 10)
        
        # Deductions
        if not vehicle_record.get('will_participate', True):
            score *= 0.3  # 70% reduction for opt-outs
        
        if vehicle_record.get('uses_public_charging', False):
            score *= 0.8  # 20% reduction (less home charging)
        
        return min(round(score, 1), 100)
    
    def calculate_operational_requirements(self, forecast_days=1):
        """
        Main function: Calculate operational requirements with UKPN zones
        """
        print(f"\n Calculating operational requirements (SOURCE-VALIDATED METHOD)...")
        print(f" Forecast horizon: {forecast_days} day(s)")
        print(f" WS1 Constraints: Min SoC {self.WS1_ABSOLUTE_MIN_SOC:.0%}, Buffer {self.WS1_RANGE_ANXIETY_BUFFER:.0%}")
        print(f" Using 5-step SoC-gap methodology")
        
        # Assign UKPN zones first
        self.fleet_df = self.assign_ukpn_zones(self.fleet_df)
        
        operational_data = []
        
        for idx, vehicle in self.fleet_df.iterrows():
            # Energy calculations (FIXED METHOD)
            energy_calc = self.calculate_required_energy(vehicle)
            
            # Time constraints
            time_calc = self.calculate_time_constraints(
                vehicle, 
                energy_calc['energy_to_charge_kwh']
            )
            
            # CP constraints check
            cp_check = self.check_cp_constraints(
                vehicle,
                energy_calc['energy_to_charge_kwh'],
                time_calc['available_charge_time_hours']
            )
            
            # Opt-out logic
            will_participate, opt_out_reason = self.apply_opt_out_logic(vehicle)
            
            # Compile operational record
            operational_record = {
                'vehicle_id': vehicle['vehicle_id'],
                'vehicle_class': vehicle['vehicle_class'],
                'make_model': vehicle['make_model'],
                
                # UKPN Zone
                'ukpn_constraint_zone': vehicle['ukpn_constraint_zone'],
                'zone_region': vehicle['zone_region'],
                'zone_priority': vehicle['zone_priority'],
                
                # Energy requirements (FIXED WITH target_soc)
                'travel_energy_kwh': energy_calc['travel_energy_kwh'],
                'public_charge_kwh': energy_calc['public_charge_kwh'],
                'home_energy_kwh': energy_calc['home_energy_kwh'],
                'required_energy_kwh': energy_calc['required_energy_kwh'],
                'energy_to_charge_kwh': energy_calc['energy_to_charge_kwh'],
                'current_soc': vehicle['return_soc'],
                'target_soc': energy_calc['target_soc'],  # NEW: target SoC
                'required_departure_soc': energy_calc['target_soc'],  # Same as target
                
                # Time constraints
                'plug_in_time': vehicle['plug_in_time'],
                'plug_out_time': vehicle['plug_out_time'],
                'available_charge_time_hours': time_calc['available_charge_time_hours'],
                'minimum_charge_time_hours': time_calc['minimum_charge_time_hours'],
                'flexibility_margin_hours': time_calc['flexibility_margin_hours'],
                'critical_latest_start': time_calc['critical_latest_start'],
                
                # Charging specs
                'effective_cp_max_kw': vehicle['effective_cp_max_kw'],
                'cp_min_kw': vehicle['cp_min_kw'],
                'required_avg_kw': cp_check['required_avg_kw'],
                'cp_constraint_violation': cp_check['cp_constraint_violation'],
                
                # Feasibility & participation
                'is_operational_feasible': time_calc['is_operational_feasible'],
                'will_participate': will_participate,
                'opt_out_reason': opt_out_reason if not will_participate else None,
                
                # Behavioral factors
                'behavioral_profile': vehicle['behavioral_profile'],
                'driver_reliability': vehicle['driver_reliability'],
                'weekday_predictability': vehicle['weekday_predictability'],
                'uses_public_charging': vehicle['uses_public_charging']
            }
            
            # Calculate flexibility score
            operational_record['flexibility_score'] = self.calculate_flexibility_score(operational_record)
            
            operational_data.append(operational_record)
        
        operational_df = pd.DataFrame(operational_data)
        
        # Summary statistics
        self._print_summary(operational_df)
        
        # Create Flexible Units aggregation
        fu_df = self._create_flexible_units(operational_df)
        
        return operational_df, fu_df
    
    def _create_flexible_units(self, operational_df):
        """
        Aggregate vehicles by UKPN zone into Flexible Units (FUs)
        """
        print(f"\n Creating Flexible Units (FUs) by UKPN Zone...")
        
        fus = []
        
        for zone in operational_df['ukpn_constraint_zone'].unique():
            zone_vehicles = operational_df[
                (operational_df['ukpn_constraint_zone'] == zone) &
                (operational_df['will_participate'] == True)
            ]
            
            if len(zone_vehicles) == 0:
                continue
            
            fu = {
                'fu_id': f"FU-{zone.replace(' ', '_')[:30]}",
                'ukpn_constraint_zone': zone,
                'zone_region': zone_vehicles.iloc[0]['zone_region'],
                'zone_priority': zone_vehicles.iloc[0]['zone_priority'],
                'num_vehicles': len(zone_vehicles),
                'total_capacity_kw': round(zone_vehicles['effective_cp_max_kw'].sum(), 2),
                'total_energy_kwh': round(zone_vehicles['energy_to_charge_kwh'].sum(), 2),
                'avg_flexibility_hours': round(zone_vehicles['flexibility_margin_hours'].mean(), 2),
                'avg_flexibility_score': round(zone_vehicles['flexibility_score'].mean(), 1),
                'cp_constraint_violations': zone_vehicles['cp_constraint_violation'].sum()
            }
            
            fus.append(fu)
        
        fu_df = pd.DataFrame(fus).sort_values('total_capacity_kw', ascending=False)
        
        print(f" Created {len(fu_df)} Flexible Units")
        print(f"\n   Top 5 FUs by Capacity:")
        for idx, fu in fu_df.head(5).iterrows():
            print(f"      {fu['fu_id'][:40]}: {fu['num_vehicles']} vehicles, {fu['total_capacity_kw']:.1f} kW")
        
        return fu_df
    
    def _print_summary(self, operational_df):
        """Print comprehensive summary"""
        print(f"\n Operational requirements calculated\n")
        
        print(f" FLEET SUMMARY:")
        print(f" Total Vehicles: {len(operational_df)}")
        print(f" Participating: {operational_df['will_participate'].sum()} ({operational_df['will_participate'].sum()/len(operational_df)*100:.1f}%)")
        print(f" Opted Out: {(~operational_df['will_participate']).sum()} ({(~operational_df['will_participate']).sum()/len(operational_df)*100:.1f}%)")
        
        # UKPN Zone distribution
        print(f"\n UKPN ZONE DISTRIBUTION:")
        zone_counts = operational_df['ukpn_constraint_zone'].value_counts()
        for zone, count in zone_counts.head(5).items():
            print(f"   {zone[:45]}: {count} vehicles")
        print(f"   ... across {len(zone_counts)} total zones")
        
        # Energy summary
        participating = operational_df[operational_df['will_participate']]
        print(f"\n ENERGY REQUIREMENTS (Participating Vehicles):")
        print(f"   Total Energy to Charge: {participating['energy_to_charge_kwh'].sum():.1f} kWh")
        print(f"   Average per Vehicle: {participating['energy_to_charge_kwh'].mean():.1f} kWh")
        print(f"   Min: {participating['energy_to_charge_kwh'].min():.1f} kWh")
        print(f"   Max: {participating['energy_to_charge_kwh'].max():.1f} kWh")
        
        # Flexibility
        print(f"\n FLEXIBILITY WINDOW:")
        print(f"   Average Flexibility Margin: {participating['flexibility_margin_hours'].mean():.1f} hours")


def calculate_fleet_operational_requirements(
    fleet_csv='data/synthetic_fleet.csv',
    output_csv='data/operational_constraints.csv',
    fu_output_csv='data/flexible_units.csv',
    forecast_days=1
):
    """
    Calculate and save operational requirements with FIXED methodology
    """
    calculator = OperationalRequirementsCalculator(fleet_csv)
    operational_df, fu_df = calculator.calculate_operational_requirements(forecast_days)
    
    # Save outputs
    operational_df.to_csv(output_csv, index=False)
    fu_df.to_csv(fu_output_csv, index=False)
    
    print(f"\n Operational constraints saved to {output_csv}")
    print(f" Flexible Units saved to {fu_output_csv}")
    
    return operational_df, fu_df


if __name__ == "__main__":
    # Run FIXED calculations
    operational_df, fu_df = calculate_fleet_operational_requirements(
        fleet_csv='data/synthetic_fleet.csv',
        output_csv='data/operational_constraints.csv',
        fu_output_csv='data/flexible_units.csv',
        forecast_days=1
    )
    
    print("\n" + "="*70)
    print(" MODULE 03 COMPLETE (SOURCE-VALIDATED FIX APPLIED)")
    print("="*70)