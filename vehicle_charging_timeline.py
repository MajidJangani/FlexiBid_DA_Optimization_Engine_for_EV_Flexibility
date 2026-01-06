"""
REFACTORED: vehicle_charging_timeline.py
=========================================
Clean separation: Computation → Plotting (2 groups)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

COLORS = {
    'reliable': '#2ECC71',
    'early_bird': '#3498DB',
    'late_arrival': '#9B59B6',
    'irregular': '#E67E22',
    'tariff_expensive': '#FF6B6B',
    'tariff_cheap': '#51CF66',
    'managed': '#3498DB',
    'unmanaged': '#E74C3C',
}


def create_vehicle_charging_analysis(baseline_csv, operational_csv, tariff_csv, num_vehicles=6):
    """
    Main computation function: calculate all metrics.
    
    NO plotting, NO printing.
    
    Args:
        baseline_csv: Path to baseline profile
        operational_csv: Path to operational constraints
        tariff_csv: Path to tariff data
        num_vehicles: Number of vehicles to analyze
    
    Returns:
        dict: {
            'vehicle_sessions': list of vehicle session dicts,
            'aggregated_metrics': aggregated load metrics,
            'tariff_by_ptu': tariff lookup dict,
            'time_labels': list of time labels
        }
    """
    # Load data
    baseline_df = pd.read_csv(baseline_csv)
    operational_df = pd.read_csv(operational_csv)
    tariff_df = pd.read_csv(tariff_csv)
    
    # Process tariff
    tariff_df['hour'] = tariff_df['time_utc'].str.split(':').str[0].astype(int)
    tariff_df['minute'] = tariff_df['time_utc'].str.split(':').str[1].astype(int)
    tariff_df['ptu'] = tariff_df['hour'] * 2 + (tariff_df['minute'] // 30)
    tariff_by_ptu = tariff_df.groupby('ptu')['value'].mean().to_dict()
    
    # Select diverse vehicles
    participating = operational_df[operational_df['will_participate'] == True].copy()
    
    selected_vehicles = []
    for profile in ['early_bird', 'reliable', 'late_arrival', 'irregular']:
        profile_vehicles = participating[participating['behavioral_profile'] == profile]
        if len(profile_vehicles) > 0:
            profile_vehicles = profile_vehicles.sort_values('plug_in_time')
            selected_vehicles.append(profile_vehicles.iloc[0])
            if len(profile_vehicles) > 1:
                selected_vehicles.append(profile_vehicles.iloc[-1])
    
    selected_vehicles = selected_vehicles[:num_vehicles]
    selected_df = pd.DataFrame(selected_vehicles)
    
    # Calculate charging sessions
    vehicle_sessions = []
    
    for idx, vehicle in selected_df.iterrows():
        h, m = map(int, vehicle['plug_in_time'].split(':'))
        plug_in_ptu = h * 2 + (1 if m >= 30 else 0)
        
        charge_rate = vehicle['effective_cp_max_kw']
        energy_needed = vehicle['energy_to_charge_kwh']
        
        # Public charging adjustment
        public_flag = False
        if vehicle['uses_public_charging'] and np.random.random() < 0.50:
            charge_rate *= 0.50
            public_flag = True
        
        hours_needed = energy_needed / charge_rate
        ptu_duration = int(np.ceil(hours_needed / 0.5))
        
        # Unmanaged charging (immediate, full power)
        vehicle_load = np.zeros(48)
        for offset in range(min(ptu_duration, 48 - plug_in_ptu)):
            ptu = (plug_in_ptu + offset) % 48
            vehicle_load[ptu] = charge_rate
        
        unmanaged_cost = sum(vehicle_load[ptu] * 0.5 * tariff_by_ptu.get(ptu, 25) / 100
                            for ptu in range(48))
        
        # Managed charging (find cheapest period)
        avg_tariffs = []
        for start_ptu in range(48):
            avg_cost = sum(tariff_by_ptu.get((start_ptu + i) % 48, 25)
                          for i in range(ptu_duration)) / ptu_duration
            avg_tariffs.append((start_ptu, avg_cost))
        
        best_start = min(avg_tariffs, key=lambda x: x[1])[0]
        
        managed_load = np.zeros(48)
        for offset in range(ptu_duration):
            ptu = (best_start + offset) % 48
            managed_load[ptu] = charge_rate
        
        managed_cost = sum(managed_load[ptu] * 0.5 * tariff_by_ptu.get(ptu, 25) / 100
                          for ptu in range(48))
        
        savings = unmanaged_cost - managed_cost
        
        vehicle_sessions.append({
            'vehicle_id': vehicle['vehicle_id'],
            'profile': vehicle['behavioral_profile'],
            'plug_in_ptu': plug_in_ptu,
            'duration': ptu_duration,
            'load': vehicle_load,
            'managed_load': managed_load,
            'charge_rate': charge_rate,
            'energy': energy_needed,
            'unmanaged_cost': unmanaged_cost,
            'managed_cost': managed_cost,
            'savings': savings,
            'public': public_flag
        })
    
    # Aggregate metrics
    total_unmanaged = sum(s['load'] for s in vehicle_sessions)
    total_managed = sum(s['managed_load'] for s in vehicle_sessions)
    
    aggregated_metrics = {
        'total_unmanaged': total_unmanaged,
        'total_managed': total_managed,
        'peak_unmanaged': np.max(total_unmanaged),
        'peak_managed': np.max(total_managed),
        'peak_reduction_kw': np.max(total_unmanaged) - np.max(total_managed),
        'peak_reduction_pct': (np.max(total_unmanaged) - np.max(total_managed)) / np.max(total_unmanaged) * 100,
        'total_savings': sum(s['savings'] for s in vehicle_sessions),
        'total_energy': sum(s['energy'] for s in vehicle_sessions)
    }
    
    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    
    return {
        'vehicle_sessions': vehicle_sessions,
        'aggregated_metrics': aggregated_metrics,
        'tariff_by_ptu': tariff_by_ptu,
        'time_labels': time_labels
    }


# ============================================================================
# PLOTTING GROUP 1: Individual Vehicle Timelines (1 large graph)
# ============================================================================

def plot_individual_vehicle_timelines(results, save_path='outputs/individual_vehicles.png', show=False):
    """
    Create 1 large graph showing individual vehicle charging timelines.
    
    NO summary table, just the timeline visualization.
    
    Args:
        results: Output from create_vehicle_charging_analysis()
        save_path: Where to save the figure
        show: Whether to display with plt.show()
    
    Returns:
        matplotlib.figure.Figure
    """
    vehicle_sessions = results['vehicle_sessions']
    tariff_by_ptu = results['tariff_by_ptu']
    time_labels = results['time_labels']
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    y_position = 0
    
    for session in vehicle_sessions:
        # Plot unmanaged (solid blocks)
        for ptu in range(48):
            if session['load'][ptu] > 0:
                rect = Rectangle((ptu, y_position), 1, 0.8,
                                facecolor=COLORS[session['profile']],
                                edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.add_patch(rect)
        
        # Plot managed (dashed blocks)
        for ptu in range(48):
            if session['managed_load'][ptu] > 0:
                rect = Rectangle((ptu, y_position + 0.15), 1, 0.4,
                                facecolor=COLORS[session['profile']],
                                edgecolor=COLORS['managed'],
                                linewidth=1.5, linestyle='--', alpha=0.4)
                ax.add_patch(rect)
        
        # Vehicle label (left side)
        label_text = f"{session['vehicle_id']}\n{session['profile'][:8]}\n{session['energy']:.1f}kWh"
        ax.text(-3, y_position + 0.4, label_text,
               fontsize=8, va='center', ha='right', fontweight='bold')
        
        # Savings (right side)
        color = 'green' if session['savings'] > 0 else 'red'
        ax.text(49, y_position + 0.4, f"£{session['savings']:.2f}",
               fontsize=9, va='center', ha='left', color=color, fontweight='bold')
        
        # Plug-in time marker
        ax.plot([session['plug_in_ptu'], session['plug_in_ptu']],
               [y_position, y_position + 0.8],
               color='red', linewidth=2, alpha=0.7)
        ax.text(session['plug_in_ptu'], y_position - 0.3,
               time_labels[session['plug_in_ptu']],
               fontsize=7, ha='center', color='red', fontweight='bold')
        
        y_position += 1.5
    
    # Background shading (tariff zones)
    for ptu in range(48):
        tariff_val = tariff_by_ptu.get(ptu, 25)
        if tariff_val > 28:
            ax.axvspan(ptu, ptu+1, alpha=0.08, color=COLORS['tariff_expensive'], zorder=0)
        elif tariff_val < 22:
            ax.axvspan(ptu, ptu+1, alpha=0.08, color=COLORS['tariff_cheap'], zorder=0)
    
    # Peak constraint window
    ax.axvspan(34, 40, alpha=0.12, color='red', zorder=0)
    ax.text(37, y_position - 0.7, 'PEAK\n17:00-20:00',
           fontsize=9, ha='center', color='red', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Axes
    ax.set_xlim(-4, 52)
    ax.set_ylim(-0.5, y_position)
    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vehicles', fontsize=12, fontweight='bold')
    ax.set_title('Individual Vehicle Charging: Unmanaged (Solid) vs Smart (Dashed)\n'
                'Red Line = Actual Plug-in Time | Savings Shown on Right',
                fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([time_labels[i] for i in range(0, 48, 4)], rotation=45)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis='x', linestyle=':')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['early_bird'], label='Early Bird'),
        mpatches.Patch(color=COLORS['reliable'], label='Reliable'),
        mpatches.Patch(color=COLORS['late_arrival'], label='Late Arrival'),
        mpatches.Patch(color=COLORS['irregular'], label='Irregular'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# ============================================================================
# PLOTTING GROUP 2: Aggregated Metrics (3 graphs)
# ============================================================================

def plot_aggregated_metrics(results, save_path='outputs/aggregated_metrics.png', show=False):
    """
    Create 3-panel aggregated metrics:
    - Left: Aggregated Load (fleet total)
    - Center: Load vs Tariff (dual y-axis)
    - Right: Cost per Vehicle (bar chart)
    
    NO summary text box.
    
    Args:
        results: Output from create_vehicle_charging_analysis()
        save_path: Where to save the figure
        show: Whether to display with plt.show()
    
    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Aggregated Load
    _plot_aggregated_load(axes[0], results)
    
    # Panel 2: Load vs Tariff
    _plot_load_vs_tariff(axes[1], results)
    
    # Panel 3: Cost per Vehicle
    _plot_cost_per_vehicle(axes[2], results)
    
    plt.suptitle(f'Aggregated Metrics: {len(results["vehicle_sessions"])} Vehicles', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def _plot_aggregated_load(ax, results):
    """Panel 1: Total fleet load"""
    agg = results['aggregated_metrics']
    ptu_range = np.arange(48)
    time_labels = results['time_labels']
    
    # Unmanaged
    ax.step(ptu_range, agg['total_unmanaged'], where='post', linewidth=3,
           color=COLORS['unmanaged'], label='Unmanaged', alpha=0.8)
    ax.fill_between(ptu_range, agg['total_unmanaged'], step='post',
                    alpha=0.3, color=COLORS['unmanaged'])
    
    # Managed
    ax.step(ptu_range, agg['total_managed'], where='post', linewidth=3,
           color=COLORS['managed'], label='Smart', alpha=0.8, linestyle='--')
    ax.fill_between(ptu_range, agg['total_managed'], step='post',
                    alpha=0.2, color=COLORS['managed'])
    
    # Peak marker
    peak_ptu = np.argmax(agg['total_unmanaged'])
    ax.scatter([peak_ptu], [agg['peak_unmanaged']], s=200,
              c='red', marker='*', edgecolors='black', linewidths=2, zorder=10)
    
    # Peak reduction annotation
    ax.annotate(f'Peak Cut:\n-{agg["peak_reduction_kw"]:.0f}kW\n({agg["peak_reduction_pct"]:.0f}%)',
                xy=(peak_ptu, agg['peak_unmanaged']),
                xytext=(peak_ptu - 8, agg['peak_unmanaged'] * 1.15),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='green', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.set_title('Aggregated Load', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time of Day', fontsize=11)
    ax.set_xticks(range(0, 48, 6))
    ax.set_xticklabels([time_labels[i] for i in range(0, 48, 6)], rotation=45, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)


def _plot_load_vs_tariff(ax, results):
    """Panel 2: Load vs tariff overlay"""
    agg = results['aggregated_metrics']
    tariff_by_ptu = results['tariff_by_ptu']
    time_labels = results['time_labels']
    ptu_range = np.arange(48)
    
    ax_twin = ax.twinx()
    
    # Load (left axis)
    ax.step(ptu_range, agg['total_unmanaged'], where='post', linewidth=2.5,
           color=COLORS['unmanaged'], label='Load', alpha=0.7)
    
    # Tariff (right axis)
    tariff_values = [tariff_by_ptu.get(ptu, 25) for ptu in range(48)]
    ax_twin.plot(ptu_range, tariff_values, color='#F39C12', linewidth=2.5,
                marker='o', markersize=3, label='Price', alpha=0.8)
    
    ax.set_title('Load vs Tariff', fontsize=11, fontweight='bold')
    ax.set_ylabel('kW', fontsize=10, color=COLORS['unmanaged'], fontweight='bold')
    ax_twin.set_ylabel('p/kWh', fontsize=10, color='#F39C12', fontweight='bold')
    ax.set_xlabel('Time of Day', fontsize=10)
    ax.set_xticks(range(0, 48, 6))
    ax.set_xticklabels([time_labels[i] for i in range(0, 48, 6)], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)
    ax_twin.legend(loc='upper right', fontsize=9)


def _plot_cost_per_vehicle(ax, results):
    """Panel 3: Cost comparison by vehicle"""
    vehicle_sessions = results['vehicle_sessions']
    
    vehicle_ids = [s['vehicle_id'] for s in vehicle_sessions]
    unmanaged_costs = [s['unmanaged_cost'] for s in vehicle_sessions]
    managed_costs = [s['managed_cost'] for s in vehicle_sessions]
    
    x = np.arange(len(vehicle_ids))
    width = 0.35
    
    ax.bar(x - width/2, unmanaged_costs, width, label='Unmanaged',
          color=COLORS['unmanaged'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, managed_costs, width, label='Smart',
          color=COLORS['managed'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Savings labels
    for i, (uc, mc) in enumerate(zip(unmanaged_costs, managed_costs)):
        ax.text(i, max(uc, mc) + max(max(unmanaged_costs), max(managed_costs)) * 0.05,
               f'£{uc-mc:.2f}', ha='center', fontsize=8, color='green', fontweight='bold')
    
    ax.set_title('Cost per Vehicle', fontsize=11, fontweight='bold')
    ax.set_ylabel('£/day', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(vehicle_ids, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Calculate all metrics (silent)
    results = create_vehicle_charging_analysis(
        baseline_csv='data/baseline_profile.csv',
        operational_csv='data/operational_constraints.csv',
        tariff_csv='data/agile_tariff.csv',
        num_vehicles=6
    )
    
    # Step 2: Plot individual vehicles (large graph)
    fig1 = plot_individual_vehicle_timelines(
        results,
        save_path='outputs/individual_vehicles.png',
        show=False
    )
    
    # Step 3: Plot aggregated metrics (3 panels)
    fig2 = plot_aggregated_metrics(
        results,
        save_path='outputs/aggregated_metrics.png',
        show=False
    )
    
    # Step 4: Access metrics for markdown report
    print("\nFor markdown report:")
    print(f"Total savings: £{results['aggregated_metrics']['total_savings']:.2f}/day")
    print(f"Peak reduction: {results['aggregated_metrics']['peak_reduction_kw']:.0f}kW ({results['aggregated_metrics']['peak_reduction_pct']:.0f}%)")
    print(f"Total energy: {results['aggregated_metrics']['total_energy']:.1f}kWh")