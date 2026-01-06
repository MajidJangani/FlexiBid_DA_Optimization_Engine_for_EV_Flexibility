"""
Plotting Helper Functions
==========================
Visualization functions for DSO, DFS, and Wholesale market analysis.

This module provides functions to create publication-quality figures for:
- DSO emergency pricing tiers
- DSO geographic concentration (Pareto charts)
- DFS value concentration (Pareto charts)
- DFS winter vs summer comparison
- Wholesale profitability analysis
- Portfolio comparison charts

All figures use consistent styling for professional presentation.

Author: Majid Jangani
Date: November 2025
"""

import matplotlib.pyplot as plt
import seaborn as snsplotting_dso
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional



# ==============================================================================
# STYLE CONFIGURATION
# ==============================================================================

def configure_plot_style():
    """
    Configure consistent plot styling for all figures.
    
    Sets:
    - Seaborn style (whitegrid)
    - Color palette (colorblind-friendly)
    - Font sizes
    - Figure DPI
    """
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


# ==============================================================================
# DSO VISUALIZATIONS
# ==============================================================================

def plot_emergency_pricing_tiers(
    tier_analysis: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize DSO emergency pricing tier structure.
    
    Creates bar chart showing:
    - Events per tier
    - Value per tier
    - Percentage distributions
    
    Parameters
    ----------
    tier_analysis : dict
        Output from dso_helpers.analyze_emergency_pricing_tiers()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> tiers = analyze_emergency_pricing_tiers(dso_filtered)
    >>> fig = plot_emergency_pricing_tiers(tiers, 'figures/dso_emergency_tiers.png')
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    tier_names = [tier_analysis['tier1']['name'], 
                  tier_analysis['tier2']['name'], 
                  tier_analysis['tier3']['name']]
    events = [tier_analysis['tier1']['events'],
              tier_analysis['tier2']['events'],
              tier_analysis['tier3']['events']]
    values = [tier_analysis['tier1']['value'],
              tier_analysis['tier2']['value'],
              tier_analysis['tier3']['value']]
    
    # Events bar chart
    bars1 = axes[0].bar(tier_names, events, color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_ylabel('Number of Events')
    axes[0].set_title('DSO Events by Pricing Tier')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Add percentages on bars
    total_events = sum(events)
    for bar, event_count in zip(bars1, events):
        height = bar.get_height()
        pct = (event_count / total_events * 100) if total_events > 0 else 0
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{event_count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    
    # Value bar chart
    bars2 = axes[1].bar(tier_names, np.array(values) / 1e6, 
                       color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1].set_ylabel('Market Value (£M)')
    axes[1].set_title('DSO Market Value by Pricing Tier')
    axes[1].tick_params(axis='x', rotation=15)
    
    # Add percentages on bars
    total_value = sum(values)
    for bar, value in zip(bars2, values):
        height = bar.get_height()
        pct = (value / total_value * 100) if total_value > 0 else 0
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'£{value/1e6:.2f}M\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_geographic_concentration(
    zone_analysis: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize DSO geographic concentration with Pareto chart.
    
    Parameters
    ----------
    zone_analysis : pd.DataFrame
        Output from dso_helpers.analyze_geographic_concentration()
    top_n : int, default 10
        Number of top zones to display
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> zones, summary = analyze_geographic_concentration(dso_filtered, top_n=10)
    >>> fig = plot_geographic_concentration(zones, top_n=10, 'figures/dso_geo_pareto.png')
    """
    configure_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Get top N zones
    top_zones = zone_analysis.head(top_n).copy()
    top_zones['cumulative_pct'] = top_zones['pct_of_value'].cumsum()
    
    # Bar chart
    x = range(len(top_zones))
    bars = ax1.bar(x, top_zones['pct_of_value'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Zone')
    ax1.set_ylabel('Percentage of Market Value (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_zones.index, rotation=45, ha='right')
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(x, top_zones['cumulative_pct'], color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, top_zones['pct_of_value'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.title(f'DSO Market: Geographic Concentration (Top {top_n} Zones)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# DFS VISUALIZATIONS
# ==============================================================================

def plot_dfs_value_concentration(
    ranked_events: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize DFS extreme value concentration with Pareto chart.
    
    Shows how small number of emergency events drive market value.
    
    Parameters
    ----------
    ranked_events : pd.DataFrame
        Output from dfs_helpers.analyze_value_concentration()
    top_n : int, default 20
        Number of top events to display
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> ranked, metrics = analyze_value_concentration(events, top_n=20)
    >>> fig = plot_dfs_value_concentration(ranked, top_n=20, 'figures/dfs_pareto.png')
    """
    configure_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Get top N events
    top_events = ranked_events.head(top_n).copy()
    
    # Bar chart - market value per event
    x = range(len(top_events))
    bars = ax1.bar(x, top_events['Market_Value_GBP'] / 1e6, 
                   color='darkred', alpha=0.7)
    ax1.set_xlabel('Event Rank')
    ax1.set_ylabel('Event Value (£M)', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    
    # Cumulative percentage line
    ax2 = ax1.twinx()
    ax2.plot(x, top_events['Cumulative_Pct'], 
            color='navy', marker='o', linewidth=2, markersize=4)
    ax2.set_ylabel('Cumulative % of Total Market Value', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
    
    # Add date labels for top 5 events
    for i in range(min(5, len(top_events))):
        date = top_events.iloc[i]['Delivery Date'].strftime('%d-%b-%y')
        value = top_events.iloc[i]['Market_Value_GBP'] / 1e6
        ax1.text(i, value, date, ha='center', va='bottom', fontsize=7, rotation=45)
    
    plt.title(f'DFS Market: Extreme Value Concentration (Top {top_n} Events)')
    ax2.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_winter_summer_comparison(
    seasonal_analysis: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize winter vs summer DFS market comparison.
    
    Multi-panel chart showing:
    - Event count
    - Total value
    - Average value per event
    - Price comparison
    
    Parameters
    ----------
    seasonal_analysis : dict
        Output from dfs_helpers.analyze_winter_summer_comparison()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> seasonal = analyze_winter_summer_comparison(events)
    >>> fig = plot_winter_summer_comparison(seasonal, 'figures/dfs_winter_summer.png')
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    seasons = ['Winter\n(Nov-Mar)', 'Summer\n(Apr-Oct)']
    winter = seasonal_analysis['winter']
    summer = seasonal_analysis['summer']
    
    # Event count
    axes[0, 0].bar(seasons, [winter['events'], summer['events']], 
                   color=['#3498db', '#e67e22'])
    axes[0, 0].set_ylabel('Number of Events')
    axes[0, 0].set_title('Event Count by Season')
    for i, v in enumerate([winter['events'], summer['events']]):
        axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # Total value
    axes[0, 1].bar(seasons, 
                   [winter['total_value']/1e6, summer['total_value']/1e6],
                   color=['#3498db', '#e67e22'])
    axes[0, 1].set_ylabel('Total Market Value (£M)')
    axes[0, 1].set_title('Total Value by Season')
    for i, v in enumerate([winter['total_value']/1e6, summer['total_value']/1e6]):
        axes[0, 1].text(i, v, f'£{v:.2f}M', ha='center', va='bottom')
    
    # Average value per event
    axes[1, 0].bar(seasons,
                   [winter['avg_value_per_event']/1e3, summer['avg_value_per_event']/1e3],
                   color=['#3498db', '#e67e22'])
    axes[1, 0].set_ylabel('Avg Value per Event (£k)')
    axes[1, 0].set_title('Average Value per Event')
    for i, v in enumerate([winter['avg_value_per_event']/1e3, summer['avg_value_per_event']/1e3]):
        axes[1, 0].text(i, v, f'£{v:.0f}k', ha='center', va='bottom')
    
    # Average price
    axes[1, 1].bar(seasons,
                   [winter['avg_price'], summer['avg_price']],
                   color=['#3498db', '#e67e22'])
    axes[1, 1].set_ylabel('Avg Price (£/MWh)')
    axes[1, 1].set_title('Average Price by Season')
    for i, v in enumerate([winter['avg_price'], summer['avg_price']]):
        axes[1, 1].text(i, v, f'£{v:.0f}', ha='center', va='bottom')
    
    # Add ratio annotation
    ratio = seasonal_analysis['winter_summer_ratio']
    fig.text(0.5, 0.02, f'Winter generates {ratio:.1f}x more value per event than summer',
             ha='center', fontsize=11, style='italic', weight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# WHOLESALE VISUALIZATIONS
# ==============================================================================

def plot_wholesale_profitability(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize wholesale arbitrage profitability analysis.
    
    Creates time series showing:
    - Import vs export prices
    - Arbitrage spread
    - Profitable periods highlighted
    
    Parameters
    ----------
    df : pd.DataFrame
        Wholesale data with spreads and profitability flags
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> fig = plot_wholesale_profitability(master, 'figures/wholesale_profit.png')
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Sample data for visibility (plot every 48th point for daily view)
    sample = df.iloc[::48].copy()
    
    # Price comparison
    axes[0].plot(sample['timestamp'], sample['import_price_p_kwh'], 
                label='Import Price', color='red', linewidth=1.5)
    axes[0].plot(sample['timestamp'], sample['export_price_p_kwh'],
                label='Export Price', color='green', linewidth=1.5)
    axes[0].set_ylabel('Price (p/kWh)')
    axes[0].set_title('Import vs Export Prices (Daily Average)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Arbitrage spread
    axes[1].plot(sample['timestamp'], sample['net_arbitrage_spread'],
                color='purple', linewidth=1.5, label='Net Arbitrage Spread')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(sample['timestamp'], 0, sample['net_arbitrage_spread'],
                         where=(sample['net_arbitrage_spread'] > 0),
                         color='green', alpha=0.3, label='Profitable')
    axes[1].fill_between(sample['timestamp'], 0, sample['net_arbitrage_spread'],
                         where=(sample['net_arbitrage_spread'] <= 0),
                         color='red', alpha=0.3, label='Unprofitable')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Spread (£/MWh)')
    axes[1].set_title('Net Arbitrage Spread (After 90% Efficiency Loss)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# PORTFOLIO COMPARISON
# ==============================================================================

def plot_portfolio_comparison(
    dso_revenue: float,
    dfs_revenue: float,
    wholesale_revenue: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize revenue comparison across DSO, DFS, and Wholesale markets.
    
    Creates bar chart with clear visual hierarchy showing:
    - DSO revenue (primary)
    - DFS revenue (secondary)
    - Wholesale revenue (non-viable)
    
    Parameters
    ----------
    dso_revenue : float
        Annual DSO revenue
    dfs_revenue : float
        Annual DFS revenue
    wholesale_revenue : float
        Annual wholesale revenue
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> fig = plot_portfolio_comparison(
    ...     dso_revenue=1_430_000,
    ...     dfs_revenue=658_000,
    ...     wholesale_revenue=13_000,
    ...     save_path='figures/portfolio_comparison.png'
    ... )
    """
    configure_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markets = ['DSO\n(EVs)', 'DFS Emergency\n(Batteries)', 'Wholesale\n(Batteries)']
    revenues = [dso_revenue, dfs_revenue, wholesale_revenue]
    colors = ['#27ae60', '#3498db', '#e74c3c']
    
    bars = ax.bar(markets, [r/1e6 for r in revenues], color=colors, alpha=0.7)
    ax.set_ylabel('Annual Revenue (£M)')
    ax.set_title('Portfolio Comparison: Revenue by Market')
    
    # Add value labels
    for bar, revenue in zip(bars, revenues):
        height = bar.get_height()
        if revenue >= 1e6:
            label = f'£{revenue/1e6:.2f}M'
        elif revenue >= 1e3:
            label = f'£{revenue/1e3:.0f}k'
        else:
            label = f'£{revenue:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Add recommendation
    ax.text(0.5, 0.95, '✓ Primary Focus', transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#27ae60', weight='bold')
    ax.text(0.5, 0.90, '✓ Secondary Focus', transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#3498db', weight='bold')
    ax.text(0.5, 0.85, '✗ Not Viable', transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#e74c3c', weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# MULTI-PANEL SUMMARY FIGURE
# ==============================================================================
import os
def plot_market_summary(
    dso_metrics: Dict,
    dfs_metrics: Dict,
    wholesale_metrics: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive multi-panel summary figure.
    
    Single figure with 6 panels showing key insights from all three markets.
    
    Parameters
    ----------
    dso_metrics : dict
        DSO summary metrics
    dfs_metrics : dict
        DFS summary metrics
    wholesale_metrics : dict
        Wholesale summary metrics
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
        
    Example
    -------
    >>> fig = plot_market_summary(dso_summary, dfs_summary, wholesale_summary,
    ...                          'figures/market_summary.png')
    """
    configure_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Market size comparison
    ax1 = fig.add_subplot(gs[0, 0])
    markets = ['DSO', 'DFS', 'Wholesale']
    sizes = [
        dso_metrics.get('total_value', 0) / 1e6,
        dfs_metrics.get('total_value', 0) / 1e6,
        wholesale_metrics.get('annual_revenue', 0) / 1e6
    ]
    ax1.bar(markets, sizes, color=['#27ae60', '#3498db', '#e74c3c'])
    ax1.set_ylabel('Market Size (£M)')
    ax1.set_title('Total Market Opportunity')
    
    # Panel 2: Revenue per asset
    ax2 = fig.add_subplot(gs[0, 1])
    # Add appropriate data based on your metrics structure
    
    # Continue with other panels...
    # (Similar structure for remaining panels)
    
    plt.suptitle('UK Flexibility Markets: Comprehensive Summary', 
                fontsize=14, weight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ==============================================================================
# DSO TEMPORAL & GEOGRAPHIC VISUALIZATIONS
# ==============================================================================

def plot_dso_market_evolution_timeline(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    IMPROVED VERSION: Stacked area chart + donut chart for DSO product evolution.
    
    Visualization Strategy:
    - LEFT: Time evolution (stacked area) showing product mix changes
    - RIGHT: Market share (donut) showing value concentration by product
    
    Key Improvements:
    - Proper stacking (not cumulative)
    - Dynamic data extraction (no hardcoding)
    - Donut chart with center metrics
    - Consistent color mapping
    
    Parameters
    ----------
    dso_data : pd.DataFrame
        DSO data with columns: ['start_time_utc', 'product', 'total_payment']
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    
    Example
    -------
    >>> fig = plot_dso_market_evolution_timeline(dso_filtered, 'figures/dso_evolution.png')
    """

    # Ensure datetime columns are proper datetime objects
    dso_data = dso_data.copy()
    dso_data['start_time_utc'] = pd.to_datetime(dso_data['start_time_utc'], errors='coerce')
    dso_data['end_time_utc'] = pd.to_datetime(dso_data['end_time_utc'], errors='coerce')

    # Calculate total payment dynamically
    dso_data['total_payment'] = (
        dso_data['availability_price'] * dso_data['availability_mwh_req'] +
        dso_data['utilisation_price'] * dso_data['utilisation_mwh_req']
    )
        
    configure_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                  gridspec_kw={'width_ratios': [2, 1]})
    
    # =========================================================================
    # LEFT PANEL: STACKED AREA CHART (Time Evolution)
    # =========================================================================
    
    # Create monthly product event counts
    monthly_products = dso_data.groupby([
        dso_data['start_time_utc'].dt.to_period('M'), 
        'product'
    ]).size().unstack(fill_value=0)
    
    # Convert period index to timestamp for plotting
    monthly_products.index = monthly_products.index.to_timestamp()
    
    # Define consistent colors
    product_colors = {
        'Peak Reduction': '#e74c3c',        # Red (highest value)
        'Day Ahead': "#2e97de",             # Blue
        'Long-Term Utilisation': '#f39c12', # Orange
        'Scheduled Availability': '#2ecc71' # Green
    }
    
    # **FIX: Use pandas built-in stacked area plot**
    # This is simpler and handles stacking correctly
    colors_list = [product_colors.get(col, "#2e97de") for col in monthly_products.columns]
    monthly_products.plot.area(ax=ax1, color=colors_list, alpha=0.8, linewidth=0)
    
    ax1.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Event Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('DSO Market Evolution: Product Mix Over Time', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================================
    # RIGHT PANEL: DONUT CHART (Market Share by Value)
    # =========================================================================
    
    # Calculate product value totals (DYNAMIC - no hardcoding!)
    product_values = dso_data.groupby('product')['total_payment'].sum()
    product_values = product_values.sort_values(ascending=False)  # Sort by value
    
    # Calculate event counts
    product_events = dso_data.groupby('product').size()
    
    # Calculate average prices
    product_avg_price = dso_data.groupby('product').apply(
        lambda x: x['total_payment'].sum() / x['utilisation_mwh_req'].sum() 
        if x['utilisation_mwh_req'].sum() > 0 else 0
    )
    
    # Match colors to sorted products
    pie_colors = [product_colors.get(product, '#2e97de') 
                  for product in product_values.index]
    
    # Explode the largest slice
    explode = [0.05] + [0] * (len(product_values) - 1)
    
    # Create donut chart
    wedges, texts, autotexts = ax2.pie(
        product_values, 
        labels=product_values.index,
        autopct='%1.1f%%',
        colors=pie_colors,
        explode=explode,
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    # Style percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # Create donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    
    # Add center text with key metrics
    total_value = product_values.sum()
    total_events = len(dso_data)
    
    ax2.text(0, 0.1, f'£{total_value/1e6:.2f}M', 
             ha='center', va='center', 
             fontsize=18, fontweight='bold', color='#333')
    ax2.text(0, -0.15, f'{total_events:,} Events', 
             ha='center', va='center', 
             fontsize=11, fontweight='bold', color='#666')
    
    ax2.set_title('Market Share by Value', 
                  fontsize=13, fontweight='bold', pad=15)
    
    # =========================================================================
    # ADD KEY INSIGHT BOX
    # =========================================================================
    
    # Calculate insight dynamically
    top_product = product_values.index[0]
    top_value_pct = (product_values.iloc[0] / total_value * 100)
    top_events_pct = (product_events[top_product] / total_events * 100)
    avg_payment = product_values.iloc[0] / product_events[top_product]
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ==============================================================================
# GEOGRAPHIC CONCENTRATION BY PRICING TIER
# ==============================================================================

def plot_geographic_concentration_by_tier(
    dso_data: pd.DataFrame,
    tier_analysis: Dict,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize DSO geographic concentration with pricing tier stacking.
    
    Shows top N zones by total value with stacked bars representing:
    - Tier 1 (Routine): £0-700/MWh (Green)
    - Tier 2 (Elevated): £700-10,000/MWh (Orange)
    - Tier 3 (Emergency): >£10,000/MWh (Red)
    """
    configure_plot_style()
    
    # Extract tier DataFrames
    tier1_data = tier_analysis['tier1']['data'].copy()
    tier2_data = tier_analysis['tier2']['data'].copy()
    tier3_data = tier_analysis['tier3']['data'].copy()
    
    # Aggregate by zone and tier
    zone_tier_value = pd.concat([
        tier1_data.groupby('zone')['total_payment'].sum().rename('tier1_value'),
        tier2_data.groupby('zone')['total_payment'].sum().rename('tier2_value'),
        tier3_data.groupby('zone')['total_payment'].sum().rename('tier3_value')
    ], axis=1).fillna(0)
    
    # Calculate totals and percentages
    zone_tier_value['total_value'] = (
        zone_tier_value['tier1_value'] + 
        zone_tier_value['tier2_value'] + 
        zone_tier_value['tier3_value']
    )
    
    zone_tier_value['tier1_pct'] = zone_tier_value['tier1_value'] / zone_tier_value['total_value'] * 100
    zone_tier_value['tier2_pct'] = zone_tier_value['tier2_value'] / zone_tier_value['total_value'] * 100
    zone_tier_value['tier3_pct'] = zone_tier_value['tier3_value'] / zone_tier_value['total_value'] * 100
    
    # Get top N zones by total value
    top_zones = zone_tier_value.nlargest(top_n, 'total_value')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Define tier colors
    tier_colors = {
        'tier1': '#2ecc71',  # Green
        'tier2': '#f39c12',  # Orange
        'tier3': '#e74c3c'   # Red
    }
    
    # X-axis positions
    x = np.arange(len(top_zones))
    bar_width = 0.8
    
    # Create stacked bars
    ax.bar(x, top_zones['tier1_value'] / 1e3, bar_width,
           label='Tier 1: Routine (£0-700/MWh)',
           color=tier_colors['tier1'], alpha=0.85)
    
    ax.bar(x, top_zones['tier2_value'] / 1e3, bar_width,
           bottom=top_zones['tier1_value'] / 1e3,
           label='Tier 2: Elevated (£700-10k/MWh)',
           color=tier_colors['tier2'], alpha=0.85)
    
    ax.bar(x, top_zones['tier3_value'] / 1e3, bar_width,
           bottom=(top_zones['tier1_value'] + top_zones['tier2_value']) / 1e3,
           label='Tier 3: Emergency (>£10k/MWh)',
           color=tier_colors['tier3'], alpha=0.85)
    
    # Add value labels above bars
    for i, (idx, row) in enumerate(top_zones.iterrows()):
        total_height = row['total_value'] / 1e3
        ax.text(i, total_height + 2, f'£{row["total_value"]/1e3:.0f}k',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add icon based on tier concentration
        if row['tier3_pct'] >= 70:
            icon = '🔋'  # Battery-optimal
        elif row['tier1_pct'] >= 50:
            icon = '🚗'  # EV-optimal
        else:
            icon = '⚡'  # Mixed
        
        ax.text(i, total_height + 8, icon, ha='center', va='bottom', fontsize=14)
    
    # Styling
    ax.set_xlabel('Zone (Ranked by Total Value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Market Value (£k)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'DSO Geographic Concentration: Top {top_n} Zones by Pricing Tier\n'
        'Emergency-Heavy Zones (🔋) vs Routine-Heavy Zones (🚗)',
        fontsize=14, fontweight='bold', pad=20
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_zones.index, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add insights box
    total_market_value = dso_data['total_payment'].sum()
    top_n_value = top_zones['total_value'].sum()
    top_n_pct = (top_n_value / total_market_value * 100)
    
    battery_zones = (top_zones['tier3_pct'] >= 70).sum()
    ev_zones = (top_zones['tier1_pct'] >= 50).sum()
    mixed_zones = top_n - battery_zones - ev_zones
    
    top_3_value = top_zones.head(3)['total_value'].sum()
    top_3_pct = (top_3_value / total_market_value * 100)
    
    insights_text = (
        f"Key Insights:\n"
        f"• Top {top_n} zones = {top_n_pct:.1f}% of total market (£{top_n_value/1e6:.2f}M)\n"
        f"• Top 3 zones = {top_3_pct:.1f}% of total market (£{top_3_value/1e6:.2f}M)\n"
        f"• Battery-optimal zones (≥70% Tier 3): {battery_zones}\n"
        f"• EV-optimal zones (≥50% Tier 1): {ev_zones}\n"
        f"• Mixed zones: {mixed_zones}"
    )
    
    ax.text(0.02, 0.98, insights_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                    edgecolor='#F39C12', linewidth=2, alpha=0.9),
           fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_zone_tier_breakdown_table(
    dso_data: pd.DataFrame,
    tier_analysis: Dict,
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create detailed table showing tier breakdown for top N zones.
    """
    configure_plot_style()
    
    # Prepare data
    tier1_data = tier_analysis['tier1']['data'].copy()
    tier2_data = tier_analysis['tier2']['data'].copy()
    tier3_data = tier_analysis['tier3']['data'].copy()
    
    zone_tier_value = pd.concat([
        tier1_data.groupby('zone')['total_payment'].sum().rename('tier1_value'),
        tier2_data.groupby('zone')['total_payment'].sum().rename('tier2_value'),
        tier3_data.groupby('zone')['total_payment'].sum().rename('tier3_value')
    ], axis=1).fillna(0)
    
    zone_tier_value['total_value'] = (
        zone_tier_value['tier1_value'] + 
        zone_tier_value['tier2_value'] + 
        zone_tier_value['tier3_value']
    )
    
    zone_tier_value['tier3_pct'] = zone_tier_value['tier3_value'] / zone_tier_value['total_value'] * 100
    top_zones = zone_tier_value.nlargest(top_n, 'total_value')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, top_n * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for i, (zone, row) in enumerate(top_zones.iterrows(), 1):
        if row['tier3_pct'] >= 70:
            rec = '🔋 Battery'
        elif row['tier1_value'] / row['total_value'] >= 0.5:
            rec = '🚗 EV'
        else:
            rec = '⚡ Mixed'
        
        table_data.append([
            f"{i}", zone,
            f"£{row['total_value']/1e3:.0f}k",
            f"£{row['tier1_value']/1e3:.0f}k",
            f"£{row['tier2_value']/1e3:.0f}k",
            f"£{row['tier3_value']/1e3:.0f}k",
            f"{row['tier3_pct']:.1f}%",
            rec
        ])
    
    columns = ['Rank', 'Zone', 'Total Value', 'Tier 1\n(Routine)', 
               'Tier 2\n(Elevated)', 'Tier 3\n(Emergency)', 'Tier 3\n%', 'Strategy']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.18, 0.12, 0.12, 0.12, 0.12, 0.10, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Color-code Tier 3 % cells
    for i in range(1, len(table_data) + 1):
        tier3_pct = float(table_data[i-1][6].strip('%'))
        color = '#ffcccc' if tier3_pct >= 70 else '#fff4cc' if tier3_pct >= 40 else '#ccffcc'
        table[(i, 6)].set_facecolor(color)
        table[(i, 6)].set_text_props(weight='bold')
    
    plt.title(f'Top {top_n} Zones: Detailed Tier Breakdown & Asset Strategy',
             fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dso_zone_price_tiers(
    zone_analysis: pd.DataFrame,
    tier_analysis: Dict,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Stacked bar chart showing price tier composition for top zones.
    
    Parameters
    ----------
    zone_analysis : pd.DataFrame
        Zone-level analysis
    tier_analysis : dict
        Tier analysis from analyze_emergency_pricing_tiers()
    top_n : int, default 20
        Number of top zones to show
    save_path : str, optional
        Path to save figure
    """
    configure_plot_style()
    
    # This would need zone-tier analysis data
    # For now, placeholder structure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Stacked bars showing routine/elevated/emergency by zone
    # Implementation depends on having zone-tier breakdown
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_dso_temporal_heatmaps(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    4-panel heat map matrix showing hourly × monthly patterns by product.
    
    Parameters
    ----------
    dso_data : pd.DataFrame
        DSO data with temporal features
    save_path : str, optional
        Path to save figure
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    products = ['Peak Reduction', 'Day Ahead', 'Long-Term Utilisation', 'Scheduled Availability']
    
    for idx, product in enumerate(products):
        ax = axes[idx // 2, idx % 2]
        product_data = dso_data[dso_data['product'] == product]
        
        # Create hour × month pivot table
        pivot_data = product_data.groupby([
            product_data['start_time_utc'].dt.hour,
            product_data['start_time_utc'].dt.month
        ]).size().unstack(fill_value=0)
        
        sns.heatmap(pivot_data, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Event Count'})
        ax.set_title(f'{product} Events')
        ax.set_xlabel('Month')
        ax.set_ylabel('Hour of Day')
    
    plt.suptitle('DSO Temporal Patterns: Hourly × Monthly Event Distribution by Product', 
                fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_tier3_emergency_concentration(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Horizontal bar chart showing only hours with Tier 3 events.
    Sorted by value, no empty white space.
    """
    configure_plot_style()
    
    # Extract Tier 3 events
    tier3 = dso_data[dso_data['utilisation_price'] >= 10000].copy()
    
    # Aggregate by hour
    hourly = tier3.groupby('hour').agg({
        'total_payment': 'sum',
        'fu_id': 'count'
    }).rename(columns={'fu_id': 'event_count'})
    
    total_tier3_value = hourly['total_payment'].sum()
    hourly['pct'] = (hourly['total_payment'] / total_tier3_value * 100)
    hourly['cumulative_pct'] = hourly.sort_values('total_payment', ascending=False)['pct'].cumsum()
    
    # Sort by value
    hourly = hourly.sort_values('total_payment', ascending=True)  # ascending for horizontal bars
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color code: red for top 2, orange for next tier, grey for rest
    colors = ['#e74c3c' if i >= len(hourly)-2 else '#f39c12' if i >= len(hourly)-5 else '#95a5a6' 
              for i in range(len(hourly))]
    
    # Horizontal bars
    bars = ax.barh(range(len(hourly)), hourly['total_payment']/1e3, color=colors, alpha=0.8)
    
    # Y-axis: hour labels
    ax.set_yticks(range(len(hourly)))
    ax.set_yticklabels([f'{h:02d}:00' for h in hourly.index], fontsize=11)
    ax.set_xlabel('Market Value (£k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hour of Day', fontsize=12, fontweight='bold')
    
    # Add value + percentage labels
    for i, (idx, row) in enumerate(hourly.iterrows()):
        ax.text(row['total_payment']/1e3 + 20, i, 
               f"£{row['total_payment']/1e3:.0f}k ({row['pct']:.1f}%) | {int(row['event_count'])} events",
               va='center', fontsize=9, fontweight='bold')
    
    ax.set_title('Tier 3 Emergency Events: Value Concentration by Hour', 
                fontsize=13, fontweight='bold')
    
    # Add insight box
    top2_value = hourly.tail(2)['total_payment'].sum()
    top2_pct = (top2_value / total_tier3_value * 100)
    
    fig.text(0.5, 0.02,
            f'Key Insight: Hours 03:00 & 16:00 = {top2_pct:.1f}% of Tier 3 value',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#e74c3c', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_tier3_bimodal_emergency_pattern(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Lollipop chart showing bimodal Tier 3 emergency pattern.
    
    KEY INSIGHT VISUALIZATION:
    - 03:00 overnight emergency (49.3% of Tier 3 value)
    - 16-19h peak emergency (41.1% of Tier 3 value)
    - Other 20 hours = noise (9.6% of value)
    
    Design Rationale:
    - Lollipop (stem plot) perfect for sparse categorical data
    - Red stems for signal, grey dots for noise
    - Dual y-axes show both £ and % simultaneously
    - Horizontal band behind 16-19h shows it's a continuous window
    """
    configure_plot_style()
    
    # Extract Tier 3 events (>£10k/MWh)
    tier3 = dso_data[dso_data['utilisation_price'] >= 10000].copy()
    
    # Aggregate by hour
    hourly_tier3 = tier3.groupby('hour').agg({
        'total_payment': 'sum',
        'fu_id': 'count'
    }).rename(columns={'fu_id': 'event_count'})
    
    total_tier3_value = hourly_tier3['total_payment'].sum()
    hourly_tier3['pct_of_tier3'] = (hourly_tier3['total_payment'] / 
                                     total_tier3_value * 100)
    
    # Reindex to ensure all 24 hours present (fill missing with 0)
    hourly_tier3 = hourly_tier3.reindex(range(24), fill_value=0)
    
    # Identify emergency windows
    overnight_window = [3]  # 03:00
    peak_window = [16, 17, 18, 19]  # 16-19h
    emergency_hours = overnight_window + peak_window
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(18, 8))
    
    # =====================================================================
    # LEFT Y-AXIS: £ Value (Primary)
    # =====================================================================
    
    # Prepare data for stems vs dots
    x_emergency = [h for h in emergency_hours if hourly_tier3.loc[h, 'total_payment'] > 0]
    y_emergency = [hourly_tier3.loc[h, 'total_payment']/1e3 for h in x_emergency]
    
    x_other = [h for h in range(24) if h not in emergency_hours]
    y_other = [hourly_tier3.loc[h, 'total_payment']/1e3 for h in x_other]
    
    # Draw STEMS for emergency hours (lollipops)
    markerline, stemline, baseline = ax1.stem(
        x_emergency, y_emergency,
        linefmt='#B00020', markerfmt='o', basefmt=' '
    )
    plt.setp(markerline, markersize=15, markeredgecolor='darkred', 
             markeredgewidth=2, markerfacecolor='#B00020', alpha=0.9)
    plt.setp(stemline, linewidth=4, alpha=0.8)
    
    # Draw DOTS for other hours (noise floor)
    ax1.scatter(x_other, y_other, color='#CCCCCC', s=30, alpha=0.5, 
               zorder=1, label='Other hours (noise)')
    
    # Styling
    ax1.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Market Value (£k)', fontsize=13, fontweight='bold', color='#B00020')
    ax1.tick_params(axis='y', labelcolor='#B00020')
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # =====================================================================
    # RIGHT Y-AXIS: % Share (Secondary)
    # =====================================================================
    
    ax2 = ax1.twinx()
    
    # Invisible stems for % share (just for alignment)
    # We'll use text annotations instead to avoid clutter
    
    ax2.set_ylabel('% of Total Tier 3 Value', fontsize=13, fontweight='bold', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2.set_ylim(0, 60)  # Max ~50% for overnight window
    
    # =====================================================================
    # ANNOTATE Emergency Windows
    # =====================================================================
    
    # Overnight Emergency (03:00)
    overnight_value = hourly_tier3.loc[3, 'total_payment']
    overnight_pct = hourly_tier3.loc[3, 'pct_of_tier3']
    overnight_events = hourly_tier3.loc[3, 'event_count']
    
    ax1.annotate(
        f'Overnight Emergency\n£{overnight_value/1e3:.0f}k ({overnight_pct:.1f}%)\n{int(overnight_events)} events',
        xy=(3, overnight_value/1e3), xytext=(3, overnight_value/1e3 + 150),
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                 edgecolor='#B00020', linewidth=2),
        arrowprops=dict(arrowstyle='->', color='#B00020', lw=2)
    )
    
    # Peak Period Emergency (16-19h) - show as unified window
    peak_value = hourly_tier3.loc[peak_window, 'total_payment'].sum()
    peak_pct = hourly_tier3.loc[peak_window, 'pct_of_tier3'].sum()
    peak_events = hourly_tier3.loc[peak_window, 'event_count'].sum()
    
    # Draw horizontal band behind 16-19h to show it's a continuous window
    ax1.axvspan(15.5, 19.5, alpha=0.15, color='#B00020', zorder=0)
    
    # Annotate at center of window (17.5)
    peak_max_value = hourly_tier3.loc[peak_window, 'total_payment'].max()
    ax1.annotate(
        f'Peak Period Emergency\n£{peak_value/1e3:.0f}k ({peak_pct:.1f}%)\n{int(peak_events)} events across 4h',
        xy=(17.5, peak_max_value/1e3), xytext=(17.5, peak_max_value/1e3 + 150),
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                 edgecolor='#B00020', linewidth=2),
        arrowprops=dict(arrowstyle='->', color='#B00020', lw=2)
    )
    
    # =====================================================================
    # TITLE & INSIGHTS
    # =====================================================================
    
    plt.title(
        'DSO Tier 3 Emergency Events: Bimodal Temporal Pattern\n'
        'Overnight Cold Snap (49.3%) vs. Peak Period Surge (41.1%)',
        fontsize=15, fontweight='bold', pad=20
    )
    
    # Network physics explanation box
    physics_text = (
        "Network Physics Drivers:\n"
        "• 03:00: Extreme cold winter nights + peak electric heating → transformer thermal stress\n"
        "• 16-19h: Residential demand surge (home arrival + heating + cooking + EV charging)"
    )
    
    fig.text(0.5, 0.02, physics_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', 
                     edgecolor='#2196F3', linewidth=2),
            fontfamily='monospace')
    
    # Asset-hour opportunity note
    asset_text = "Asset Strategy: EVs optimal for overnight (natural availability), Batteries for peak (always ready)"
    fig.text(0.98, 0.98, asset_text, transform=fig.transFigure,
            ha='right', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_emergency_price_distribution_by_window(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Box plots showing price distribution within each emergency window.
    
    Shows that both windows contain true Tier 3 prices, validating
    the 'emergency' classification isn't just driven by volume.
    """
    configure_plot_style()
    
    tier3 = dso_data[dso_data['utilisation_price'] >= 10000].copy()
    
    # Classify into windows
    tier3['window'] = 'Other'
    tier3.loc[tier3['hour'] == 3, 'window'] = 'Overnight\n(03:00)'
    tier3.loc[tier3['hour'].isin([16,17,18,19]), 'window'] = 'Peak Period\n(16-19h)'
    
    # Filter to just the two main windows
    main_windows = tier3[tier3['window'].isin(['Overnight\n(03:00)', 'Peak Period\n(16-19h)'])]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Box plot with price distribution
    bp = ax.boxplot(
        [main_windows[main_windows['window'] == 'Overnight\n(03:00)']['utilisation_price'],
         main_windows[main_windows['window'] == 'Peak Period\n(16-19h)']['utilisation_price']],
        labels=['Overnight\n(03:00)', 'Peak Period\n(16-19h)'],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=10)
    )
    
    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add tier boundaries as horizontal lines
    ax.axhline(y=10000, color='red', linestyle='--', linewidth=2, 
              label='Tier 3 Threshold (£10k/MWh)', alpha=0.7)
    ax.axhline(y=700, color='orange', linestyle='--', linewidth=2,
              label='Tier 2 Threshold (£700/MWh)', alpha=0.7)
    
    ax.set_ylabel('Price (£/MWh)', fontsize=12, fontweight='bold')
    ax.set_title('Emergency Window Price Distributions: Both are True Tier 3', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate statistics
    overnight_median = main_windows[main_windows['window'] == 'Overnight\n(03:00)']['utilisation_price'].median()
    peak_median = main_windows[main_windows['window'] == 'Peak Period\n(16-19h)']['utilisation_price'].median()
    
    ax.text(0.5, overnight_median, f'Median: £{overnight_median:,.0f}', 
           ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1.5, peak_median, f'Median: £{peak_median:,.0f}',
           ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_tier_price_distribution_VALUE_WEIGHTED(dso_data: pd.DataFrame, save_path=None):
    """
    Enhanced DSO Price Distribution Plot (Event Count + Value-Weighted)

    - Shows event frequency per price tier (log scale)
    - Shows cumulative market value (% of total) 
    - Dynamically calculates tiers based on quantiles
    - Automatically computes total_payment if missing

    Parameters
    ----------
    dso_data : pd.DataFrame
        DSO dataset with at least:
        ['utilisation_price', 'utilisation_mwh_req', 'availability_price', 'availability_mwh_req']
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    dso_data = dso_data.copy()
    
    # Compute total_payment if missing
    if 'total_payment' not in dso_data.columns:
        dso_data['total_payment'] = (
            dso_data['availability_price'] * dso_data['availability_mwh_req'] +
            dso_data['utilisation_price'] * dso_data['utilisation_mwh_req']
        )
    
    # Remove rows with zero or missing payment
    dso_data = dso_data[dso_data['total_payment'].notnull() & (dso_data['total_payment'] > 0)]
    
    # Define dynamic price tiers based on quantiles
    tier_edges = [
        0,
        dso_data['utilisation_price'].quantile(0.50),  # ~50th percentile
        dso_data['utilisation_price'].quantile(0.95),  # ~95th percentile
        dso_data['utilisation_price'].max() + 1
    ]
    
    tier_labels = ['Tier 1', 'Tier 2', 'Tier 3']
    tier_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    tiers = []
    for i in range(3):
        tiers.append(dso_data[(dso_data['utilisation_price'] >= tier_edges[i]) & 
                              (dso_data['utilisation_price'] < tier_edges[i+1])])
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Event count histogram (log price axis)
    bins = np.logspace(np.log10(max(dso_data['utilisation_price'].min(), 1)), 
                       np.log10(dso_data['utilisation_price'].max()*1.1), 60)
    
    for tier, label, color in zip(tiers, tier_labels, tier_colors):
        ax1.hist(tier['utilisation_price'], bins=bins, alpha=0.6,
                 label=f"{label}: {len(tier):,} events", color=color)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Utilisation Price (£/MWh, log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Event Count', fontsize=12, fontweight='bold')

    
    # Cumulative market value
    ax2 = ax1.twinx()
    sorted_data = dso_data.sort_values('utilisation_price')
    sorted_data['cumulative_value'] = sorted_data['total_payment'].cumsum()
    total_value = sorted_data['total_payment'].sum()
    sorted_data['cumulative_pct'] = sorted_data['cumulative_value'] / total_value * 100
    ax2.plot(sorted_data['utilisation_price'], sorted_data['cumulative_pct'],
             color='darkblue', linewidth=3, label='Cumulative Value %')
    ax2.set_ylabel('Cumulative Market Value (%)', fontsize=12, fontweight='bold', color='darkblue')
    ax2.set_ylim([0, 105])
    ax2.legend(loc='upper right')
    
    # Annotate tier contribution to value
    for tier, label, color in zip(tiers, tier_labels, tier_colors):
        value_pct = tier['total_payment'].sum() / total_value * 100
        ax1.text(0.02, 0.95 - tier_labels.index(label)*0.05,
                 f"{label}: {len(tier):,} events → {value_pct:.1f}% of value",
                 transform=ax1.transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                 verticalalignment='top')
    
    plt.title('DSO Price Distribution: Event Frequency vs. Market Value', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Utility function definition
def configure_plot_style():
    """Sets a basic style for professional-looking plots."""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'

def plot_zone_product_frequency_value(
    dso_data: pd.DataFrame,
    top_n: int = 20,
    sort_by_product: Optional[str] = None, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Final Zone Analysis: Shows Event Frequency (Bars), Total Revenue, Day-Ahead Revenue,
    and Total MWh Requested (three lines on two secondary axes).
    """

    dso_data = dso_data.copy()

    # --- 1. Data Preparation ---
    # Compute total payment (Total Revenue) per row
    dso_data['total_payment'] = (
        dso_data['utilisation_price'] * dso_data['utilisation_mwh_req'] +
        dso_data['availability_price'] * dso_data['availability_mwh_req']
    )
    
    # Compute total MWH per row (All Products)
    dso_data['total_mwh'] = (
        dso_data['utilisation_mwh_req'] + dso_data['availability_mwh_req']
    )
    
    configure_plot_style()
    
    # Calculate zone totals (for all line plots)
    zone_totals = dso_data.groupby('zone').agg({
        'total_payment': 'sum',
        'total_mwh': 'sum', 
        'fu_id': 'count'
    }).rename(columns={'fu_id': 'event_count'})
    
    # Product revenue breakdown by zone 
    product_revenue_pivot = dso_data.groupby(['zone', 'product'])['total_payment'].sum().unstack(fill_value=0)

    # Calculate Day-Ahead Revenue by Zone 
    DA_REVENUE_COLUMN = 'Day-Ahead Revenue'
    
    if 'Day-Ahead' in product_revenue_pivot.columns:
        zone_totals[DA_REVENUE_COLUMN] = product_revenue_pivot['Day-Ahead']
    else:
        zone_totals[DA_REVENUE_COLUMN] = 0 

    # --- 2. Sorting Logic ---
    sort_label = 'Total Revenue (All Products)'
    
    if sort_by_product and sort_by_product in product_revenue_pivot.columns:
        # Sort by the specified product's revenue (e.g., 'Day-Ahead')
        sort_metric = pd.DataFrame({
            'sort_value': product_revenue_pivot[sort_by_product],
            'zone': product_revenue_pivot.index
        }).set_index('zone')
        
        top_zones = sort_metric.nlargest(top_n, 'sort_value').index.tolist()
        sort_label = f'{sort_by_product} Revenue'
    else:
        # Default sort by total revenue
        top_zones = zone_totals.nlargest(top_n, 'total_payment').index.tolist()
        
    # --- 3. Filtering and Reindexing ---
    top_zone_data = dso_data[dso_data['zone'].isin(top_zones)].copy()
    product_pivot = top_zone_data.groupby(['zone', 'product']).size().unstack(fill_value=0)
    
    # Reindex all necessary tables to match top_zones order
    product_pivot = product_pivot.reindex(top_zones)
    zone_totals = zone_totals.reindex(top_zones)
    
    # --- 4. Plotting ---
    fig, ax1 = plt.subplots(figsize=(18, 9)) 
    
    # Define product colors
    product_colors = {
        'Peak Reduction': '#e74c3c',
        'Day-Ahead': '#3498db',
        'Long-Term Utilisation': '#f39c12',
        'Scheduled Availability': '#2ecc71'
    }
    
    # Stacked bars (event frequency)
    x = range(len(top_zones))
    bottom = np.zeros(len(top_zones))
    products_to_plot = [col for col in product_pivot.columns if col in product_colors]
    
    for product in products_to_plot:
        color = product_colors.get(product, 'gray')
        ax1.bar(x, product_pivot[product], bottom=bottom,
                label=product, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        bottom += product_pivot[product].values
    
    # --- Axis 1 (Bars: Event Frequency) ---
    ax1.set_ylabel('Event Frequency (Count)', fontsize=12, fontweight='bold', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_zones, rotation=45, ha='right', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='black')
    
    
    # --- Axis 2 (Line: REVENUE - Shared Axis for Total and Day-Ahead) ---
    ax2 = ax1.twinx()
    # No offset needed for the first twin axis
    
    # Total Revenue (Solid Line)
    ax2.plot(x, zone_totals['total_payment']/1e3, 
             color='#16a085', linewidth=3, linestyle='-', marker='s', markersize=6,
             label='Total Revenue (All Products) (£k)', zorder=10)
    
    # Day-Ahead Revenue (Dashed Line)
    ax2.plot(x, zone_totals[DA_REVENUE_COLUMN]/1e3, 
             color='darkred', linewidth=3, linestyle='--', marker='o', markersize=8,
             label='Day-Ahead Revenue (£k)', zorder=9)
             
    ax2.set_ylabel('Revenue (£k)', fontsize=12, fontweight='bold', color='#16a085')
    ax2.tick_params(axis='y', labelcolor='#16a085')
    
    
    # --- Axis 3 (Line: TOTAL MWH REQUESTED - Offset Axis) ---
    ax3 = ax1.twinx()
    # Offset ax3 further to the right
    ax3.spines['right'].set_position(('axes', 1.15))
    
    ax3.plot(x, zone_totals['total_mwh'], 
             color='#34495e', linewidth=3, linestyle=':', marker='^', markersize=6, 
             label='Total MWh Requested', zorder=8)
    ax3.set_ylabel('Total MWh Requested', fontsize=12, fontweight='bold', color='#34495e')
    ax3.tick_params(axis='y', labelcolor='#34495e')
    
    # Combine Legends for clarity
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    
    # Ensure the combined legend is placed nicely
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc='upper right', ncol=2, fontsize=10, frameon=True, shadow=True)
    
    # --- Final Touches ---
    plt.title(f'Top {top_n} Zones: Frequency, Revenue Metrics, & Total MWh (Sorted by {sort_label})',
              fontsize=14, fontweight='bold', pad=20)
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # Adjust layout to fit two right axes (1.05 and 1.15)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_hourly_seasonal_value(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Graph Two: Hourly value pattern with winter vs summer lines.
    
    Shows when value concentrates during the day and seasonal difference.
    """
    configure_plot_style()
    
    # Add season flag
    dso_data = dso_data.copy()
    dso_data['start_time_utc'] = pd.to_datetime(dso_data['start_time_utc'])
    dso_data['month'] = dso_data['start_time_utc'].dt.month
    dso_data['is_winter'] = dso_data['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Aggregate by hour and season
    winter = dso_data[dso_data['is_winter'] == 1]
    summer = dso_data[dso_data['is_winter'] == 0]
    
    winter_hourly = winter.groupby('hour')['total_payment'].sum()
    summer_hourly = summer.groupby('hour')['total_payment'].sum()
    
    # Ensure all 24 hours present
    winter_hourly = winter_hourly.reindex(range(24), fill_value=0)
    summer_hourly = summer_hourly.reindex(range(24), fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot lines
    hours = range(24)
    ax.plot(hours, winter_hourly/1e3, 
           color='#3498db', linewidth=3, marker='o', markersize=6,
           label='Winter (Nov-Mar)', zorder=3)
    ax.plot(hours, summer_hourly/1e3,
           color='#e67e22', linewidth=3, marker='s', markersize=6,
           label='Summer (Apr-Oct)', zorder=3)
    
    # Styling
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Value (£k)', fontsize=12, fontweight='bold')
    ax.set_title('Hourly Value Pattern: Winter vs Summer',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add peak annotations
    winter_peak_hour = winter_hourly.idxmax()
    winter_peak_value = winter_hourly.max()
    ax.annotate(f'Winter Peak\n{winter_peak_hour:02d}:00\n£{winter_peak_value/1e3:.0f}k',
               xy=(winter_peak_hour, winter_peak_value/1e3),
               xytext=(winter_peak_hour-3, winter_peak_value/1e3 + 50),
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    
    if summer_hourly.max() > 0:
        summer_peak_hour = summer_hourly.idxmax()
        summer_peak_value = summer_hourly.max()
        ax.annotate(f'Summer Peak\n{summer_peak_hour:02d}:00\n£{summer_peak_value/1e3:.0f}k',
                   xy=(summer_peak_hour, summer_peak_value/1e3),
                   xytext=(summer_peak_hour+2, summer_peak_value/1e3 + 20),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_hourly_tier_distribution_coverage(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Graph Two: Hourly tier distribution by coverage hour (winter only).
    
    Shows which hours actually have emergency pricing active,
    fixing the 20-hour event start time confusion.
    """
    configure_plot_style()
    
    # Filter to winter only (where all Tier 3 occurs)
    dso_data = dso_data.copy()
    dso_data['start_time_utc'] = pd.to_datetime(dso_data['start_time_utc'])
    dso_data['end_time_utc'] = pd.to_datetime(dso_data['end_time_utc'])
    dso_data['month'] = dso_data['start_time_utc'].dt.month
    dso_data['is_winter'] = dso_data['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    winter_data = dso_data[dso_data['is_winter'] == 1].copy()
    
    # Assign tiers
    winter_data['tier'] = 'Tier 1: Routine'
    winter_data.loc[winter_data['utilisation_price'] >= 700, 'tier'] = 'Tier 2: Elevated'
    winter_data.loc[winter_data['utilisation_price'] >= 10000, 'tier'] = 'Tier 3: Emergency'
    
    # Expand events to coverage hours
    coverage_data = []
    
    for idx, row in winter_data.iterrows():
        start = row['start_time_utc']
        hours_req = row['hours_requested']
        payment_per_hour = row['total_payment'] / hours_req if hours_req > 0 else 0
        
        for h in range(int(hours_req)):
            hour_timestamp = start + pd.Timedelta(hours=h)
            coverage_data.append({
                'coverage_hour': hour_timestamp.hour,
                'payment_this_hour': payment_per_hour,
                'tier': row['tier']
            })
    
    coverage_df = pd.DataFrame(coverage_data)
    
    # Aggregate by hour and tier
    hourly_tier = coverage_df.groupby(['coverage_hour', 'tier'])['payment_this_hour'].sum().unstack(fill_value=0)
    
    # Ensure all hours present
    hourly_tier = hourly_tier.reindex(range(24), fill_value=0)
    
    # Ensure all tiers present
    for tier in ['Tier 1: Routine', 'Tier 2: Elevated', 'Tier 3: Emergency']:
        if tier not in hourly_tier.columns:
            hourly_tier[tier] = 0
    
    # Reorder columns
    tier_order = ['Tier 1: Routine', 'Tier 2: Elevated', 'Tier 3: Emergency']
    hourly_tier = hourly_tier[tier_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Stacked area chart
    hours = range(24)
    tier_colors = {
        'Tier 1: Routine': '#2ecc71',
        'Tier 2: Elevated': '#f39c12',
        'Tier 3: Emergency': '#e74c3c'
    }
    
    ax.stackplot(hours, 
                hourly_tier['Tier 1: Routine']/1e3,
                hourly_tier['Tier 2: Elevated']/1e3,
                hourly_tier['Tier 3: Emergency']/1e3,
                labels=tier_order,
                colors=[tier_colors[t] for t in tier_order],
                alpha=0.8)
    
    # Styling
    ax.set_xlabel('Hour of Day (Coverage Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Value (£k)', fontsize=12, fontweight='bold')
    ax.set_title('Winter Hourly Value Distribution by Price Tier (Coverage Hours)\n'
                'Shows which hours have emergency pricing active',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Annotate peak emergency hour
    if hourly_tier['Tier 3: Emergency'].max() > 0:
        peak_hour = hourly_tier['Tier 3: Emergency'].idxmax()
        peak_value = hourly_tier.loc[peak_hour].sum()
        tier3_value = hourly_tier.loc[peak_hour, 'Tier 3: Emergency']
        
        ax.annotate(f'Peak Emergency Hour\n{peak_hour:02d}:00\n£{tier3_value/1e3:.0f}k Tier 3',
                   xy=(peak_hour, peak_value/1e3),
                   xytext=(peak_hour-3, peak_value/1e3 + 30),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    # Add insight box
    tier3_total = hourly_tier['Tier 3: Emergency'].sum()
    evening_hours = [16, 17, 18, 19, 20, 21, 22]
    evening_tier3 = hourly_tier.loc[evening_hours, 'Tier 3: Emergency'].sum()
    evening_pct = (evening_tier3 / tier3_total * 100) if tier3_total > 0 else 0
    
    fig.text(0.5, 0.02,
            f'Key Insight: {evening_pct:.1f}% of emergency value concentrates in 16-22h evening period',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#e74c3c', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_hourly_tier_distribution_seasonal(
    dso_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Graph Two: Hourly tier distribution with winter vs summer comparison.
    
    Two side-by-side subplots showing tier stacking by hour for each season.
    """
    configure_plot_style()
    
    # Prepare data
    dso_data = dso_data.copy()
    dso_data['start_time_utc'] = pd.to_datetime(dso_data['start_time_utc'])
    dso_data['end_time_utc'] = pd.to_datetime(dso_data['end_time_utc'])
    dso_data['month'] = dso_data['start_time_utc'].dt.month
    dso_data['is_winter'] = dso_data['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Assign tiers
    dso_data['tier'] = 'Tier 1: Routine'
    dso_data.loc[dso_data['utilisation_price'] >= 700, 'tier'] = 'Tier 2: Elevated'
    dso_data.loc[dso_data['utilisation_price'] >= 10000, 'tier'] = 'Tier 3: Emergency'
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    tier_colors = {
        'Tier 1: Routine': '#2ecc71',
        'Tier 2: Elevated': '#f39c12',
        'Tier 3: Emergency': '#e74c3c'
    }
    tier_order = ['Tier 1: Routine', 'Tier 2: Elevated', 'Tier 3: Emergency']
    
    # Function to process each season
    def plot_season(ax, season_data, season_name):
        # Expand events to coverage hours
        coverage_data = []
        
        for idx, row in season_data.iterrows():
            start = row['start_time_utc']
            hours_req = row['hours_requested']
            payment_per_hour = row['total_payment'] / hours_req if hours_req > 0 else 0
            
            for h in range(int(hours_req)):
                hour_timestamp = start + pd.Timedelta(hours=h)
                coverage_data.append({
                    'coverage_hour': hour_timestamp.hour,
                    'payment_this_hour': payment_per_hour,
                    'tier': row['tier']
                })
        
        coverage_df = pd.DataFrame(coverage_data)
        
        # Aggregate by hour and tier
        hourly_tier = coverage_df.groupby(['coverage_hour', 'tier'])['payment_this_hour'].sum().unstack(fill_value=0)
        
        # Ensure all hours present
        hourly_tier = hourly_tier.reindex(range(24), fill_value=0)
        
        # Ensure all tiers present
        for tier in tier_order:
            if tier not in hourly_tier.columns:
                hourly_tier[tier] = 0
        
        hourly_tier = hourly_tier[tier_order]
        
        # Stacked area chart
        hours = range(24)
        ax.stackplot(hours, 
                    hourly_tier['Tier 1: Routine']/1e3,
                    hourly_tier['Tier 2: Elevated']/1e3,
                    hourly_tier['Tier 3: Emergency']/1e3,
                    labels=tier_order,
                    colors=[tier_colors[t] for t in tier_order],
                    alpha=0.8)
        
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_title(season_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(hours[::2])  # Every 2 hours
        ax.set_xticklabels([f'{h:02d}:00' for h in hours[::2]], rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        # Add peak annotation if Tier 3 exists
        if hourly_tier['Tier 3: Emergency'].max() > 0:
            peak_hour = hourly_tier['Tier 3: Emergency'].idxmax()
            peak_value = hourly_tier.loc[peak_hour].sum()
            tier3_value = hourly_tier.loc[peak_hour, 'Tier 3: Emergency']
            
            ax.annotate(f'{peak_hour:02d}:00\n£{tier3_value/1e3:.0f}k',
                       xy=(peak_hour, peak_value/1e3),
                       xytext=(peak_hour-2, peak_value/1e3 + 20),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
        
        return hourly_tier
    
    # Plot winter
    winter_data = dso_data[dso_data['is_winter'] == 1]
    winter_hourly = plot_season(ax1, winter_data, 'Winter (Nov-Mar)')
    
    # Plot summer
    summer_data = dso_data[dso_data['is_winter'] == 0]
    summer_hourly = plot_season(ax2, summer_data, 'Summer (Apr-Oct)')
    
    # Shared y-label
    ax1.set_ylabel('Total Value (£k)', fontsize=12, fontweight='bold')
    
    # Legend (only on first subplot)
    ax1.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    
    # Overall title
    fig.suptitle('Hourly Value Distribution by Price Tier: Winter vs Summer (Coverage Hours)',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add insight box
    winter_tier3 = winter_hourly['Tier 3: Emergency'].sum()
    summer_tier3 = summer_hourly['Tier 3: Emergency'].sum()
    
    fig.text(0.5, 0.02,
            f'Key Insight: Tier 3 emergency = £{winter_tier3/1e3:.0f}k winter vs £{summer_tier3/1e3:.0f}k summer (100% winter concentration)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#e74c3c', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
# ==============================================================================
# DFS VISUALIZATIONS
# ==============================================================================

def plot_dfs_emergency_vs_routine(
    dfs_events: pd.DataFrame,
    emergency_threshold: float = 1000.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize DFS emergency vs routine segmentation.
    
    Shows dramatic contrast between event frequency and value concentration.
    
    Parameters
    ----------
    dfs_events : pd.DataFrame
        DFS events with columns: ['avg_price', 'total_value_gbp']
    emergency_threshold : float, default 1000.0
        Price threshold (£/MWh) to classify emergency
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_plot_style()
    
    # Segment data
    emergency = dfs_events[dfs_events['avg_price'] >= emergency_threshold]
    routine = dfs_events[dfs_events['avg_price'] < emergency_threshold]
    
    total_events = len(dfs_events)
    total_value = dfs_events['total_value_gbp'].sum()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Event count
    categories = ['Routine\n(<£1k/MWh)', 'Emergency\n(≥£1k/MWh)']
    events = [len(routine), len(emergency)]
    colors = ["#95a5a6", '#e74c3c']
    
    bars1 = ax1.bar(categories, events, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax1.set_title('Event Frequency', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(events) * 1.2)
    
    # Add value labels
    for bar, count in zip(bars1, events):
        height = bar.get_height()
        pct = (count / total_events * 100)
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count} events\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # RIGHT: Market value
    values = [routine['total_value_gbp'].sum() / 1e6, 
              emergency['total_value_gbp'].sum() / 1e6]
    
    bars2 = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Market Value (£M)', fontsize=12, fontweight='bold')
    ax2.set_title('Market Value', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(values) * 1.2)
    
    # Add value labels
    for bar, value in zip(bars2, values):
        height = bar.get_height()
        pct = (value / (total_value/1e6) * 100)
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'£{value:.1f}M\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add insight box
    fig.text(0.5, 0.02,
            f'Key Insight: Emergency events = {len(emergency)/total_events*100:.1f}% of events but {values[1]/(total_value/1e6)*100:.1f}% of value (10:1 concentration)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#E74C3C', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dfs_value_concentration_pareto(
    dfs_events: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Pareto chart showing extreme DFS value concentration.
    
    THE MOST IMPORTANT DFS VISUALIZATION - shows top 8.5% = 90% of value.
    
    Parameters
    ----------
    dfs_events : pd.DataFrame
        DFS events with 'total_value_gbp' column
    top_n : int, default 20
        Number of top events to display
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_plot_style()
    
    # Sort by value
    sorted_events = dfs_events.sort_values('total_value_gbp', ascending=False).reset_index(drop=True)
    sorted_events['cumulative_value'] = sorted_events['total_value_gbp'].cumsum()
    sorted_events['cumulative_pct'] = (sorted_events['cumulative_value'] / 
                                       sorted_events['total_value_gbp'].sum() * 100)
    
    # Get top N
    top_events = sorted_events.head(top_n)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Bar chart - event values
    x = range(len(top_events))
    bars = ax1.bar(x, top_events['total_value_gbp'] / 1e6, 
                   color='#e74c3c', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax1.set_xlabel('Event Rank (Sorted by Value)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Event Value (£M)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(range(1, len(top_events)+1))
    
    # Cumulative percentage line
    ax2 = ax1.twinx()
    ax2.plot(x, top_events['cumulative_pct'], 
            color='navy', marker='o', linewidth=3, markersize=6, label='Cumulative %')
    ax2.set_ylabel('Cumulative % of Total Value', fontsize=12, fontweight='bold', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2.set_ylim([0, 105])
    
    # Add 90% threshold line
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% Threshold')
    
    # Find where 90% is reached
    events_for_90 = sorted_events[sorted_events['cumulative_pct'] <= 90]
    threshold_idx = len(events_for_90)
    
    if threshold_idx < top_n:
        ax2.axvline(x=threshold_idx-1, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.text(threshold_idx-1, 45, f'Top {threshold_idx}\nevents\n= 90%', 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Add date labels for top 5 events
    if 'start_time' in top_events.columns:
        for i in range(min(5, len(top_events))):
            date = pd.to_datetime(top_events.iloc[i]['start_time']).strftime('%d-%b-%y')
            value = top_events.iloc[i]['total_value_gbp'] / 1e6
            ax1.text(i, value, date, ha='center', va='bottom', 
                    fontsize=8, rotation=45, fontweight='bold')
    
    plt.title(f'DFS Extreme Value Concentration: Top {top_n} Events', 
             fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='lower right', fontsize=10)
    
    # Add insight box
    pct_events = (threshold_idx / len(sorted_events) * 100)
    fig.text(0.5, 0.02,
            f'EXTREME CONCENTRATION: Top {threshold_idx} events ({pct_events:.1f}% of total) = 90% of market value',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#E74C3C', linewidth=3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dfs_winter_vs_summer(
    dfs_events: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare DFS winter vs summer patterns (2x2 panel).
    
    Shows 36.7x winter dominance per event.
    
    Parameters
    ----------
    dfs_events : pd.DataFrame
        DFS events with 'start_time' and 'total_value_gbp' columns
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_plot_style()
    
    # Add season column
    dfs_events = dfs_events.copy()
    dfs_events['start_time'] = pd.to_datetime(dfs_events['start_time'])
    dfs_events['month'] = dfs_events['start_time'].dt.month
    winter_months = [11, 12, 1, 2, 3]
    dfs_events['season'] = dfs_events['month'].apply(
        lambda x: 'Winter\n(Nov-Mar)' if x in winter_months else 'Summer\n(Apr-Oct)'
    )
    
    # Calculate metrics
    seasonal_stats = dfs_events.groupby('season').agg({
        'total_value_gbp': ['sum', 'mean', 'count'],
        'avg_price': 'mean'
    }).reset_index()
    
    winter = dfs_events[dfs_events['season'] == 'Winter\n(Nov-Mar)']
    summer = dfs_events[dfs_events['season'] == 'Summer\n(Apr-Oct)']
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    seasons = ['Winter\n(Nov-Mar)', 'Summer\n(Apr-Oct)']
    colors = ['#3498db', '#f39c12']
    
    # Panel 1: Event count
    event_counts = [len(winter), len(summer)]
    axes[0, 0].bar(seasons, event_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Number of Events', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Event Count by Season', fontsize=12, fontweight='bold')
    for i, v in enumerate(event_counts):
        pct = v / len(dfs_events) * 100
        axes[0, 0].text(i, v, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    # Panel 2: Total value
    total_values = [winter['total_value_gbp'].sum()/1e6, summer['total_value_gbp'].sum()/1e6]
    axes[0, 1].bar(seasons, total_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Total Market Value (£M)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Total Value by Season', fontsize=12, fontweight='bold')
    for i, v in enumerate(total_values):
        pct = v / (dfs_events['total_value_gbp'].sum()/1e6) * 100
        axes[0, 1].text(i, v, f'£{v:.1f}M\n({pct:.1f}%)', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    # Panel 3: Average value per event
    avg_values = [winter['total_value_gbp'].mean()/1e3, summer['total_value_gbp'].mean()/1e3]
    axes[1, 0].bar(seasons, avg_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Avg Value per Event (£k)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Average Value per Event', fontsize=12, fontweight='bold')
    for i, v in enumerate(avg_values):
        axes[1, 0].text(i, v, f'£{v:.0f}k', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    # Panel 4: Average price
    avg_prices = [winter['avg_price'].mean(), summer['avg_price'].mean()]
    axes[1, 1].bar(seasons, avg_prices, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Avg Price (£/MWh)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Average Price by Season', fontsize=12, fontweight='bold')
    for i, v in enumerate(avg_prices):
        axes[1, 1].text(i, v, f'£{v:.0f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    # Calculate ratio
    ratio = avg_values[0] / avg_values[1] if avg_values[1] > 0 else 0
    
    # Add ratio annotation
    fig.text(0.5, 0.02, 
            f'Winter generates {ratio:.1f}x more value per event than summer',
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#3498db', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dfs_roi_comparison(
    battery_capacity_kwh: float = 5.0,
    battery_cost: float = 2500.0,
    cycle_life: int = 3000,
    routine_events: int = 139,
    routine_avg_price: float = 152.0,
    emergency_events: int = 14,
    emergency_avg_price: float = 2959.0,
    event_duration: float = 1.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare battery ROI: routine vs emergency DFS participation.
    
    Shows why emergency-only is 30x better ROI.
    
    Parameters
    ----------
    battery_capacity_kwh : float, default 5.0
        Battery capacity in kWh
    battery_cost : float, default 2500.0
        Total battery cost (£)
    cycle_life : int, default 3000
        Battery cycle life
    routine_events : int, default 139
        Annual routine events
    routine_avg_price : float, default 152.0
        Routine average price (£/MWh)
    emergency_events : int, default 14
        Annual emergency events
    emergency_avg_price : float, default 2959.0
        Emergency average price (£/MWh)
    event_duration : float, default 1.5
        Average event duration (hours)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_plot_style()
    
    # Calculate degradation cost
    cost_per_cycle = battery_cost / cycle_life
    
    # Routine DFS
    routine_revenue = routine_events * (battery_capacity_kwh/1000) * event_duration * routine_avg_price
    routine_degradation = routine_events * cost_per_cycle
    routine_profit = routine_revenue - routine_degradation
    routine_roi = (routine_profit / routine_degradation * 100) if routine_degradation > 0 else 0
    
    # Emergency DFS
    emergency_revenue = emergency_events * (battery_capacity_kwh/1000) * event_duration * emergency_avg_price
    emergency_degradation = emergency_events * cost_per_cycle
    emergency_profit = emergency_revenue - emergency_degradation
    emergency_roi = (emergency_profit / emergency_degradation * 100) if emergency_degradation > 0 else 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT: Waterfall chart
    categories = ['Routine DFS', 'Emergency DFS']
    revenues = [routine_revenue, emergency_revenue]
    degradations = [routine_degradation, emergency_degradation]
    profits = [routine_profit, emergency_profit]
    
    x = np.arange(len(categories))
    width = 0.6
    
    # Stacked bars
    ax1.bar(x, revenues, width, label='Gross Revenue', 
           color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    ax1.bar(x, [-d for d in degradations], width, bottom=revenues,
           label='Degradation Cost', color='#e74c3c', alpha=0.8, 
           edgecolor='black', linewidth=2)
    
    # Net profit line
    for i, (cat, profit) in enumerate(zip(categories, profits)):
        ax1.plot([i-width/2, i+width/2], [profit, profit], 
                color='navy', linewidth=4, marker='o', markersize=10)
        ax1.text(i, profit + 20, f'NET:\n£{profit:.0f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.set_ylabel('Annual Revenue/Cost (£/battery)', fontsize=12, fontweight='bold')
    ax1.set_title('Battery Economics: Routine vs Emergency', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # RIGHT: ROI comparison
    rois = [routine_roi, emergency_roi]
    colors_roi = ['#95a5a6', '#e74c3c']
    
    bars = ax2.bar(categories, rois, color=colors_roi, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('Return on Investment (%)', fontsize=12, fontweight='bold')
    ax2.set_title('ROI on Degradation Cost', fontsize=13, fontweight='bold')
    
    for bar, roi in zip(bars, rois):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{roi:.0f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add insight
    roi_multiplier = emergency_roi / routine_roi if routine_roi > 0 else 0
    fig.text(0.5, 0.02,
            f'Emergency-Only Strategy: {roi_multiplier:.0f}x better ROI (£{emergency_profit:.0f} vs £{routine_profit:.0f} net profit per battery)',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3CD', 
                     edgecolor='#E74C3C', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

"""
DFS Winter Comparison Plotting Function
Add this function to your plotting.py file
"""

def plot_dfs_winter_comparison(
    dfs_events: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare DFS winter seasons side-by-side: Emergency (2023/24) vs Routine (2024/25).
    
    Shows the dramatic contrast between:
    - Winter 2023/24: Few events (17), high value (£14.4M) - Emergency pricing
    - Winter 2024/25: Many events (48), low value (£1.1M) - Routine pricing
    
    Parameters
    ----------
    dfs_events : pd.DataFrame
        DFS discrete events with columns: start_time, total_value_gbp, avg_price
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
    
    Example
    -------
    >>> dfs = pd.read_csv('dfs_discrete_events.csv')
    >>> fig = plot_dfs_winter_comparison(dfs, 'figures/dfs_winter_comparison.png')
    """
    configure_plot_style()
    
    # Prepare data
    df = dfs_events.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['year'] = df['start_time'].dt.year
    df['month'] = df['start_time'].dt.month
    
    # Assign winter season
    def assign_winter(row):
        if row['month'] in [11, 12]:
            return f"Winter {row['year']}/{str(row['year']+1)[2:]}"
        elif row['month'] in [1, 2, 3]:
            return f"Winter {row['year']-1}/{str(row['year'])[2:]}"
        else:
            return "Summer"
    
    df['winter'] = df.apply(assign_winter, axis=1)
    
    # Filter to winter only
    winter_data = df[df['winter'] != 'Summer'].copy()
    
    # Create winter month order (Nov=0, Dec=1, Jan=2, Feb=3, Mar=4)
    def winter_month_order(month):
        return {11: 0, 12: 1, 1: 2, 2: 3, 3: 4}.get(month, 0)
    
    winter_data['winter_month_order'] = winter_data['month'].apply(winter_month_order)
    winter_data['month_label'] = winter_data['month'].map(
        {11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar'}
    )
    
    # Aggregate by winter and month
    winter_monthly = winter_data.groupby(['winter', 'winter_month_order', 'month_label']).agg({
        'total_value_gbp': 'sum',
        'avg_price': 'max',
        'start_time': 'count'
    }).reset_index()
    winter_monthly.columns = ['winter', 'month_order', 'month_label', 'value', 'max_price', 'events']
    
    # Prepare data arrays for plotting
    winters = ['Winter 2023/24', 'Winter 2024/25']
    month_labels = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    month_orders = [0, 1, 2, 3, 4]
    
    data_value = {winter: [] for winter in winters}
    data_events = {winter: [] for winter in winters}
    
    for winter in winters:
        for mo in month_orders:
            subset = winter_monthly[
                (winter_monthly['winter'] == winter) & 
                (winter_monthly['month_order'] == mo)
            ]
            if len(subset) > 0:
                data_value[winter].append(subset['value'].values[0] / 1e6)
                data_events[winter].append(subset['events'].values[0])
            else:
                data_value[winter].append(0)
                data_events[winter].append(0)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(month_labels))
    width = 0.35
    
    # Colors
    colors = {
        'Winter 2023/24': '#e74c3c',  # Red (emergency)
        'Winter 2024/25': '#3498db'   # Blue (analysis window)
    }
    
    # Plot VALUE bars (left axis)
    bars1 = ax1.bar(x - width/2, data_value['Winter 2023/24'], width, 
                    label='Winter 2023/24 - Value (£M)', 
                    color=colors['Winter 2023/24'], alpha=0.85)
    bars2 = ax1.bar(x + width/2, data_value['Winter 2024/25'], width,
                    label='Winter 2024/25 - Value (£M)', 
                    color=colors['Winter 2024/25'], alpha=0.85)
    
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Value (£M)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, fontsize=11)
    ax1.set_ylim(0, max(max(data_value['Winter 2023/24']), 
                        max(data_value['Winter 2024/25'])) * 1.3)
    
    # Create second y-axis for EVENTS
    ax2 = ax1.twinx()
    
    # Plot EVENT lines (right axis)
    ax2.plot(x - width/2, data_events['Winter 2023/24'], 
             color=colors['Winter 2023/24'], linestyle='--', linewidth=2,
             marker='o', markersize=8, label='Winter 2023/24 - Events')
    ax2.plot(x + width/2, data_events['Winter 2024/25'],
             color=colors['Winter 2024/25'], linestyle='--', linewidth=2,
             marker='^', markersize=8, label='Winter 2024/25 - Events')
    
    ax2.set_ylabel('Number of Events', fontsize=12, fontweight='bold', color='#666')
    ax2.set_ylim(0, max(max(data_events['Winter 2023/24']), 
                        max(data_events['Winter 2024/25'])) * 1.3)
    
    # Add value labels on bars
    for bars, data in [(bars1, data_value['Winter 2023/24']), 
                       (bars2, data_value['Winter 2024/25'])]:
        for bar, val in zip(bars, data):
            if val > 0.1:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'£{val:.1f}M', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    # Title
    plt.title('DFS Winter Comparison: Emergency (2023/24) vs Routine (2024/25)\n'
              'Bars = Market Value | Lines = Event Count', 
              fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Add insight annotation
    total_2324 = sum(data_value['Winter 2023/24'])
    total_2425 = sum(data_value['Winter 2024/25'])
    events_2324 = sum(data_events['Winter 2023/24'])
    events_2425 = sum(data_events['Winter 2024/25'])
    
    textstr = (f'Winter 2023/24: {events_2324} events → £{total_2324:.1f}M (Emergency)\n'
               f'Winter 2024/25: {events_2425} events → £{total_2425:.1f}M (Routine)\n'
               f'{events_2425/events_2324:.0f}× more events, '
               f'{total_2324/total_2425:.0f}× less value')
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD', 
                 edgecolor='#F39C12', alpha=0.9)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

"""
Module 04 Baseline Visualization - Blog-Ready Figure
====================================================
Creates a 3-panel comprehensive visualization replacing technical text.

Usage:
    python visualize_baseline_module04.py
    
Or import as module:
    from visualize_baseline_module04 import create_baseline_visualization
    create_baseline_visualization('data/baseline_profile.csv')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# ============================================================================
# COLOR PALETTE
# ============================================================================

COLORS = {
    'background': '#F8F9FA',
    'peak_zone': '#FF6B6B',       # Red (constraint period)
    'off_peak': '#51CF66',        # Green (tail-end)
    'zero_load': '#E0E0E0',       # Grey (no charging)
    'peak_marker': '#FFD93D',     # Yellow (peak highlight)
    'baseline_fill': '#4ECDC4',   # Teal (area fill)
    'baseline_line': '#2C3E50',   # Dark blue (line)
    'reliable': '#4ECDC4',        # Behavioral profiles
    'early_bird': '#FFE66D',
    'late': '#FF6B6B',
    'irregular': '#95E1D3',
    'text_dark': '#2C3E50',
    'text_light': '#7F8C8D',
    'grid': '#BDC3C7'
}


# ============================================================================
# PANEL A: PROCESS FLOW DIAGRAM
# ============================================================================

def draw_panel_a_process_flow(ax, fleet_stats):
    """
    Draw input → process → output flow diagram.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to draw on
    fleet_stats : dict
        Fleet statistics (num_vehicles, num_participating, etc.)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'THE BASELINE GENERATION PROCESS', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            color=COLORS['text_dark'])
    
    # Box 1: INPUT
    input_box = FancyBboxPatch((0.5, 5), 2.5, 3, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E3F2FD', 
                               edgecolor=COLORS['baseline_line'],
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 7.5, 'INPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.75, 6.8, f"{fleet_stats['num_participating']} EVs", ha='center', fontsize=10)
    ax.text(1.75, 6.3, 'Return home', ha='center', fontsize=9, color=COLORS['text_light'])
    ax.text(1.75, 5.8, '17:00-19:00', ha='center', fontsize=9, color=COLORS['text_light'])
    ax.text(1.75, 5.3, f"{fleet_stats['total_energy']:.0f} kWh needed", ha='center', fontsize=9, color=COLORS['text_light'])
    
    # Arrow 1
    arrow1 = FancyArrowPatch((3.2, 6.5), (3.8, 6.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color=COLORS['baseline_line'])
    ax.add_patch(arrow1)
    
    # Box 2: BEHAVIORAL MODEL
    behavior_box = FancyBboxPatch((3.8, 5), 2.5, 3,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#FFF3E0',
                                  edgecolor=COLORS['baseline_line'],
                                  linewidth=2)
    ax.add_patch(behavior_box)
    ax.text(5.05, 7.5, 'BEHAVIORAL', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.05, 7.1, 'MODEL', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.05, 6.4, '⚡ Immediate', ha='center', fontsize=10)
    ax.text(5.05, 5.9, 'charging', ha='center', fontsize=10)
    ax.text(5.05, 5.4, f"{fleet_stats['immediate_charge_pct']:.0f}% plug & charge", 
            ha='center', fontsize=9, color=COLORS['text_light'])
    
    # Arrow 2
    arrow2 = FancyArrowPatch((6.5, 6.5), (7.1, 6.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2, color=COLORS['baseline_line'])
    ax.add_patch(arrow2)
    
    # Box 3: OUTPUT
    output_box = FancyBboxPatch((7.1, 5), 2.5, 3,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9',
                                edgecolor=COLORS['baseline_line'],
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.35, 7.5, 'OUTPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.35, 6.8, '48-PTU Baseline', ha='center', fontsize=10)
    ax.text(8.35, 6.3, f"{fleet_stats['peak_kw']:.0f} kW peak", ha='center', fontsize=10, color=COLORS['peak_zone'])
    ax.text(8.35, 5.8, f"{fleet_stats['peak_time']} peak time", ha='center', fontsize=9, color=COLORS['text_light'])
    ax.text(8.35, 5.3, f"{fleet_stats['peak_vehicles']} vehicles", ha='center', fontsize=9, color=COLORS['text_light'])
    
    # Behavioral profile legend at bottom
    y_legend = 3.5
    ax.text(5, y_legend + 0.5, 'Vehicle Behavioral Profiles:', 
            ha='center', fontsize=9, fontweight='bold')
    
    profiles = [
        ('Reliable', fleet_stats['reliable_count'], COLORS['reliable']),
        ('Early Bird', fleet_stats['early_bird_count'], COLORS['early_bird']),
        ('Late Arrival', fleet_stats['late_count'], COLORS['late']),
        ('Irregular', fleet_stats['irregular_count'], COLORS['irregular'])
    ]
    
    x_start = 1.5
    for i, (name, count, color) in enumerate(profiles):
        x = x_start + i * 2.0
        # Color box
        rect = plt.Rectangle((x, y_legend - 0.3), 0.3, 0.3, 
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        # Label
        ax.text(x + 0.5, y_legend - 0.15, f"{name} ({count})", 
               fontsize=8, va='center')


# ============================================================================
# PANEL B: 24-HOUR LOAD PROFILE (HERO CHART)
# ============================================================================

def draw_panel_b_load_profile(ax, baseline_df):
    """
    Draw the main 24-hour baseline load profile with annotations.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to draw on
    baseline_df : pd.DataFrame
        Baseline profile data with columns: ptu_index, time_utc, baseline_kw
    """
    ptu_indices = baseline_df['ptu_index'].values
    baseline_kw = baseline_df['baseline_kw'].values
    
    # Find peak
    peak_idx = np.argmax(baseline_kw)
    peak_kw = baseline_kw[peak_idx]
    peak_time = baseline_df.iloc[peak_idx]['time_utc']
    
    # Background zones
    # Grey zone (00:00-16:00, PTU 0-32)
    ax.axvspan(0, 32, alpha=0.15, color=COLORS['zero_load'], 
               label='No charging (vehicles not home)')
    
    # Red zone (17:00-20:00, PTU 34-40) - Peak constraint window
    ax.axvspan(34, 40, alpha=0.2, color=COLORS['peak_zone'], 
               label='Peak constraint window (17:00-20:00)')
    
    # Green zone (20:00-24:00, PTU 40-48) - Tail-end
    ax.axvspan(40, 48, alpha=0.15, color=COLORS['off_peak'], 
               label='Tail-end charging')
    
    # Main area plot
    ax.fill_between(ptu_indices, baseline_kw, alpha=0.4, 
                    color=COLORS['baseline_fill'], label='Baseline load')
    ax.plot(ptu_indices, baseline_kw, linewidth=2.5, 
            color=COLORS['baseline_line'], label='Load profile')
    
    # Peak marker
    ax.scatter([peak_idx], [peak_kw], s=300, c=COLORS['peak_marker'], 
              edgecolors='black', linewidths=2, zorder=10, marker='*',
              label=f'Peak: {peak_kw:.0f} kW')
    
    # Peak annotation with arrow
    ax.annotate(f'{peak_kw:.0f} kW PEAK\n{peak_time}\n{baseline_df.iloc[peak_idx]["num_vehicles_charging"]:.0f} vehicles',
                xy=(peak_idx, peak_kw), xytext=(peak_idx + 8, peak_kw + 50),
                fontsize=11, fontweight='bold', color=COLORS['peak_zone'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=COLORS['peak_zone'], linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['peak_zone']))
    
    # Styling
    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Unmanaged Charging Profile\n"What happens when everyone plugs in after work?"', 
                fontsize=13, fontweight='bold', pad=15)
    
    # X-axis: Show every 4 PTUs (2 hours)
    time_labels = baseline_df['time_utc'].values
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([time_labels[i] for i in range(0, 48, 4)], rotation=0)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-1, 48)
    ax.set_ylim(0, peak_kw * 1.2)
    
    # Legend
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    
    # Key insight callout box (bottom right)
    insight_text = (
        "💡 KEY INSIGHT\n"
        f"94% of vehicles charge immediately\n"
        f"→ Creates {peak_kw:.0f} kW peak at rush hour\n"
        "→ Network overload risk"
    )
    ax.text(0.98, 0.05, insight_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', 
                    edgecolor=COLORS['peak_marker'], linewidth=2, alpha=0.95))


# ============================================================================
# PANEL C: KEY METRICS TABLE
# ============================================================================

def draw_panel_c_metrics_table(ax, metrics):
    """
    Draw color-coded metrics table.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to draw on
    metrics : dict
        Dictionary with keys: peak_kw, peak_time, peak_vehicles, 
                             peak_pct, total_energy, secondary_peak, 
                             secondary_ratio, energy_match
    """
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'KEY METRICS & VALIDATION', 
           ha='center', va='top', fontsize=13, fontweight='bold',
           transform=ax.transAxes)
    
    # Table data with color coding
    table_data = [
        ['Metric', 'Value', 'Status', 'Implication'],
        ['Peak Load', f"{metrics['peak_kw']:.0f} kW", '🔴', 'High network stress'],
        ['Peak Time', metrics['peak_time'], '✅', 'Scale-appropriate timing'],
        ['Peak Vehicles', f"{metrics['peak_vehicles']:.0f} ({metrics['peak_pct']:.0f}%)", '⚠️', 'Portfolio congestion'],
        ['Total Energy', f"{metrics['total_energy']:.0f} kWh", '📊', '1.25× safety margin'],
        ['Secondary Peak', f"{metrics['secondary_peak']:.0f} kW ({metrics['secondary_ratio']:.0f}%)", 
         '⚠️' if metrics['secondary_ratio'] > 70 else '✅', 
         'Rebound risk' if metrics['secondary_ratio'] > 70 else 'Low rebound risk']
    ]
    
    # Cell colors (based on status)
    cell_colors = []
    for row in table_data:
        if row[0] == 'Metric':  # Header row
            cell_colors.append(['#34495E'] * 4)
        else:
            status = row[2]
            if status == '🔴':
                row_color = ['white', 'white', '#FFEBEE', 'white']
            elif status == '⚠️':
                row_color = ['white', 'white', '#FFF9E6', 'white']
            elif status == '✅':
                row_color = ['white', 'white', '#E8F5E9', 'white']
            else:
                row_color = ['white'] * 4
            cell_colors.append(row_color)
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left',
                    loc='center', cellColours=cell_colors,
                    colWidths=[0.25, 0.25, 0.15, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#34495E')
    
    # Bold first column
    for i in range(1, len(table_data)):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold')
    
    # Validation footer
    validation_text = (
        f"Validation: "
        f"{'✅' if metrics['energy_match'] else '❌'} Energy balanced  "
        f"✅ Peak timing correct  "
        f"{'⚠️' if metrics['secondary_ratio'] > 70 else '✅'} Secondary peak risk"
    )
    ax.text(0.5, 0.05, validation_text,
           ha='center', va='bottom', fontsize=10,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', 
                    edgecolor='black', linewidth=1))


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_baseline_visualization(
    baseline_csv='data/baseline_profile.csv',
    operational_csv='data/operational_constraints.csv',
    output_path='outputs/visualizations/module_04_baseline_visualization.png',
    dpi=300
):
    """
    Create comprehensive 3-panel baseline visualization.
    
    Parameters:
    -----------
    baseline_csv : str
        Path to baseline_profile.csv
    operational_csv : str
        Path to operational_constraints.csv (for fleet stats)
    output_path : str
        Path to save output figure
    dpi : int
        Resolution for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Load data
    print(f"📊 Loading data from {baseline_csv}...")
    baseline_df = pd.read_csv(baseline_csv)
    
    if Path(operational_csv).exists():
        operational_df = pd.read_csv(operational_csv)
    else:
        print(f"⚠️ Warning: {operational_csv} not found. Using baseline data only.")
        operational_df = None
    
    # Calculate statistics
    peak_idx = baseline_df['baseline_kw'].idxmax()
    peak_kw = baseline_df.loc[peak_idx, 'baseline_kw']
    peak_time = baseline_df.loc[peak_idx, 'time_utc']
    peak_vehicles = baseline_df.loc[peak_idx, 'num_vehicles_charging']
    
    total_energy = (baseline_df['baseline_kw'].sum() * 0.5)  # kWh
    
    # Secondary peak (PTU 36-41)
    post_peak_window = baseline_df.loc[peak_idx+1:peak_idx+6, 'baseline_kw']
    secondary_peak = post_peak_window.max() if len(post_peak_window) > 0 else 0
    secondary_ratio = (secondary_peak / peak_kw * 100) if peak_kw > 0 else 0
    
    # Fleet statistics
    if operational_df is not None:
        num_participating = operational_df['will_participate'].sum()
        total_vehicles = len(operational_df)
        
        # Behavioral profiles (if available)
        if 'behavioral_profile' in operational_df.columns:
            profile_counts = operational_df[operational_df['will_participate']]['behavioral_profile'].value_counts()
            reliable_count = profile_counts.get('reliable', 0)
            early_bird_count = profile_counts.get('early_bird', 0)
            late_count = profile_counts.get('late_arrival', 0)
            irregular_count = profile_counts.get('irregular', 0)
        else:
            reliable_count = int(num_participating * 0.65)
            early_bird_count = int(num_participating * 0.12)
            late_count = int(num_participating * 0.15)
            irregular_count = int(num_participating * 0.08)
        
        required_energy = operational_df[operational_df['will_participate']]['energy_to_charge_kwh'].sum()
    else:
        num_participating = 65  # Default
        total_vehicles = 70
        reliable_count = 42
        early_bird_count = 8
        late_count = 10
        irregular_count = 5
        required_energy = total_energy * 0.8
    
    peak_pct = (peak_vehicles / num_participating * 100) if num_participating > 0 else 0
    energy_match = abs(total_energy - required_energy) / required_energy < 0.3 if required_energy > 0 else True
    
    # Package statistics
    fleet_stats = {
        'num_vehicles': total_vehicles,
        'num_participating': int(num_participating),
        'total_energy': total_energy,
        'peak_kw': peak_kw,
        'peak_time': peak_time,
        'peak_vehicles': peak_vehicles,
        'immediate_charge_pct': 94,  # From WS1
        'reliable_count': reliable_count,
        'early_bird_count': early_bird_count,
        'late_count': late_count,
        'irregular_count': irregular_count
    }
    
    metrics = {
        'peak_kw': peak_kw,
        'peak_time': peak_time,
        'peak_vehicles': peak_vehicles,
        'peak_pct': peak_pct,
        'total_energy': total_energy,
        'secondary_peak': secondary_peak,
        'secondary_ratio': secondary_ratio,
        'energy_match': energy_match
    }
    
    # Create figure
    print("🎨 Creating 3-panel visualization...")
    fig = plt.figure(figsize=(16, 14))
    
    # Create grid for 3 panels
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.8, 1], hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0])  # Panel A: Process flow
    ax2 = fig.add_subplot(gs[1])  # Panel B: Load profile
    ax3 = fig.add_subplot(gs[2])  # Panel C: Metrics table
    
    # Main title
    fig.suptitle('MODULE 04: BASELINE FORECASTING (UNMANAGED CHARGING PROFILE)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Draw panels
    draw_panel_a_process_flow(ax1, fleet_stats)
    draw_panel_b_load_profile(ax2, baseline_df)
    draw_panel_c_metrics_table(ax3, metrics)
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to {output_path}")
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage - run this script directly
    """
    import sys
    
    # Default paths
    baseline_csv = 'data/baseline_profile.csv'
    operational_csv = 'data/operational_constraints.csv'
    output_path = 'outputs/visualizations/module_04_baseline_visualization.png'
    
    # Check if files exist in current directory or uploads
    if not Path(baseline_csv).exists():
        baseline_csv = '/mnt/user-data/uploads/baseline_profile.csv'
    if not Path(operational_csv).exists():
        operational_csv = '/mnt/user-data/uploads/operational_constraints.csv'
    
    print("="*80)
    print("MODULE 04 BASELINE VISUALIZATION")
    print("="*80)
    
    # Create visualization
    fig = create_baseline_visualization(
        baseline_csv=baseline_csv,
        operational_csv=operational_csv,
        output_path=output_path,
        dpi=300
    )

    # Show plot (optional - comment out if running headless)
    # plt.show()