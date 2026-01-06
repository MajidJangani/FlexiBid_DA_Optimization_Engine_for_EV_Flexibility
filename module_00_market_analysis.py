# module_00_market_analysis.py
"""
UKPN Day-Ahead Flexibility Market Analysis
===========================================
Comprehensive analysis of 2,981 flexibility events to extract:
- Competitor strategies (focus on Axle Energy benchmark)
- Pricing dynamics and distribution
- Temporal patterns (peak demand windows)
- Geographic insights

Data Source: UKPN flexibility tender data (Nov-Dec 2025)
Scope: Educational benchmarking for portfolio project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class UKPNMarketAnalyzer:
    """
    Analyzes UKPN Day-Ahead flexibility market data to extract
    competitive intelligence and market dynamics.
    """
    
    def __init__(self, filepath):
        """
        Initialize analyzer with UKPN data
        
        Parameters:
        -----------
        filepath : str
            Path to ukpnflexibilitydemandturndown.csv
        """
        self.df = self._load_and_clean_data(filepath)
        self.analysis_results = {}
        
    def _load_and_clean_data(self, filepath):
        """Load and preprocess UKPN data"""
        df = pd.read_csv(filepath)
        
        # Convert timestamps
        df['start_time_utc'] = pd.to_datetime(df['start_time_utc'], utc=True)
        df['end_time_utc'] = pd.to_datetime(df['end_time_utc'], utc=True)
        
        # Extract temporal features
        df['start_hour'] = df['start_time_utc'].dt.hour
        df['day_of_week'] = df['start_time_utc'].dt.day_name()
        df['month'] = df['start_time_utc'].dt.month
        df['season'] = df['month'].map({
            11: 'Autumn', 12: 'Winter', 1: 'Winter', 2: 'Winter'
        })
        
        # Calculate revenue
        df['revenue'] = df['utilisation_mwh_req'] * df['utilisation_price']
        
        # Calculate duration in hours
        df['duration_hours'] = (df['end_time_utc'] - df['start_time_utc']).dt.total_seconds() / 3600
        
        print(f"✅ Loaded {len(df)} events from UKPN dataset")
        print(f"   Date range: {df['start_time_utc'].min()} to {df['start_time_utc'].max()}")
        print(f"   Total market value: £{df['revenue'].sum():,.2f}")
        
        return df
    
    def analyze_market_overview(self):
        """Generate high-level market statistics"""
        overview = {
            'total_events': len(self.df),
            'total_revenue': self.df['revenue'].sum(),
            'total_energy_mwh': self.df['utilisation_mwh_req'].sum(),
            'avg_price_per_mwh': self.df['utilisation_price'].mean(),
            'median_price_per_mwh': self.df['utilisation_price'].median(),
            'avg_event_duration_hours': self.df['duration_hours'].mean(),
            'unique_companies': self.df['company_name'].nunique(),
            'unique_zones': self.df['zone'].nunique()
        }
        
        self.analysis_results['market_overview'] = overview
        return overview
    
    def analyze_competitor_landscape(self):
        """Detailed competitor analysis with market share"""
        competitors = self.df.groupby('company_name').agg({
            'revenue': 'sum',
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'utilisation_price': 'mean',
            'duration_hours': 'mean'
        }).rename(columns={
            'fu_id': 'num_events',
            'utilisation_price': 'avg_price_per_mwh'
        }).round(2)
        
        # Calculate market share
        competitors['market_share_pct'] = (
            competitors['revenue'] / competitors['revenue'].sum() * 100
        ).round(1)
        
        # Sort by revenue
        competitors = competitors.sort_values('revenue', ascending=False)
        
        # Calculate revenue per event
        competitors['revenue_per_event'] = (
            competitors['revenue'] / competitors['num_events']
        ).round(2)
        
        self.analysis_results['competitors'] = competitors
        return competitors
    
    def analyze_technology_breakdown(self):
        """Compare performance across technology types"""
        tech_analysis = self.df.groupby('technology').agg({
            'revenue': 'sum',
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'utilisation_price': 'mean',
            'duration_hours': 'mean'
        }).rename(columns={
            'fu_id': 'num_events',
            'utilisation_price': 'avg_price_per_mwh'
        }).round(2)
        
        tech_analysis = tech_analysis.sort_values('revenue', ascending=False)
        
        self.analysis_results['technology'] = tech_analysis
        return tech_analysis
    
    def analyze_temporal_patterns(self):
        """Extract time-based demand patterns"""
        # Hourly distribution
        hourly = self.df.groupby('start_hour').agg({
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'revenue': 'sum'
        }).rename(columns={'fu_id': 'num_events'})
        
        # Day of week distribution
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = self.df.groupby('day_of_week').agg({
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'revenue': 'sum'
        }).rename(columns={'fu_id': 'num_events'})
        daily = daily.reindex(day_order)
        
        # Seasonal distribution
        seasonal = self.df.groupby('season').agg({
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'revenue': 'sum'
        }).rename(columns={'fu_id': 'num_events'})
        
        self.analysis_results['temporal'] = {
            'hourly': hourly,
            'daily': daily,
            'seasonal': seasonal
        }
        
        return hourly, daily, seasonal
    
    def analyze_pricing_dynamics(self):
        """Detailed pricing analysis"""
        # Overall distribution
        price_stats = {
            'min': self.df['utilisation_price'].min(),
            'q25': self.df['utilisation_price'].quantile(0.25),
            'median': self.df['utilisation_price'].median(),
            'q75': self.df['utilisation_price'].quantile(0.75),
            'max': self.df['utilisation_price'].max(),
            'mean': self.df['utilisation_price'].mean(),
            'std': self.df['utilisation_price'].std()
        }
        
        # Price by technology
        price_by_tech = self.df.groupby('technology')['utilisation_price'].agg([
            'mean', 'median', 'min', 'max', 'count'
        ]).round(2)
        
        # Price vs duration correlation
        price_duration_corr = self.df[['utilisation_price', 'duration_hours']].corr().iloc[0, 1]
        
        self.analysis_results['pricing'] = {
            'statistics': price_stats,
            'by_technology': price_by_tech,
            'duration_correlation': price_duration_corr
        }
        
        return price_stats, price_by_tech
    
    def analyze_geographic_distribution(self):
        """Top zones by revenue and activity"""
        zones = self.df.groupby('zone').agg({
            'revenue': 'sum',
            'utilisation_mwh_req': 'sum',
            'fu_id': 'count',
            'utilisation_price': 'mean'
        }).rename(columns={
            'fu_id': 'num_events',
            'utilisation_price': 'avg_price_per_mwh'
        }).round(2)
        
        zones = zones.sort_values('revenue', ascending=False)
        
        self.analysis_results['zones'] = zones
        return zones
    
    def analyze_axle_strategy(self):
        """
        Deep dive into Axle Energy as market leader benchmark
        """
        axle_data = self.df[self.df['company_name'] == 'Axle Energy Limited']
        
        if len(axle_data) == 0:
            print("⚠️  No Axle Energy data found")
            return None
        
        axle_strategy = {
            'total_events': len(axle_data),
            'total_revenue': axle_data['revenue'].sum(),
            'market_share_pct': (axle_data['revenue'].sum() / self.df['revenue'].sum() * 100),
            'avg_price': axle_data['utilisation_price'].mean(),
            'median_price': axle_data['utilisation_price'].median(),
            'price_range': (axle_data['utilisation_price'].min(), axle_data['utilisation_price'].max()),
            'avg_duration_hours': axle_data['duration_hours'].mean(),
            'avg_volume_mwh': axle_data['utilisation_mwh_req'].mean(),
            'unique_zones': axle_data['zone'].nunique(),
            'peak_hour': axle_data['start_hour'].mode()[0] if len(axle_data['start_hour'].mode()) > 0 else None
        }
        
        # Top zones for Axle
        axle_zones = axle_data.groupby('zone').agg({
            'revenue': 'sum',
            'fu_id': 'count'
        }).rename(columns={'fu_id': 'num_events'}).sort_values('revenue', ascending=False).head(10)
        
        axle_strategy['top_zones'] = axle_zones
        
        self.analysis_results['axle_strategy'] = axle_strategy
        return axle_strategy
    
    def generate_comprehensive_report(self):
        """
        Run all analyses and return consolidated results
        """
        print("\n" + "="*70)
        print("🔍 UKPN DAY-AHEAD FLEXIBILITY MARKET ANALYSIS")
        print("="*70)
        
        # Run all analyses
        overview = self.analyze_market_overview()
        competitors = self.analyze_competitor_landscape()
        tech = self.analyze_technology_breakdown()
        hourly, daily, seasonal = self.analyze_temporal_patterns()
        price_stats, price_by_tech = self.analyze_pricing_dynamics()
        zones = self.analyze_geographic_distribution()
        axle = self.analyze_axle_strategy()
        
        # Print key findings
        print(f"\n📊 MARKET OVERVIEW")
        print(f"   Total Events: {overview['total_events']:,}")
        print(f"   Total Revenue: £{overview['total_revenue']:,.2f}")
        print(f"   Average Price: £{overview['avg_price_per_mwh']:.2f}/MWh")
        print(f"   Median Price: £{overview['median_price_per_mwh']:.2f}/MWh")
        
        print(f"\n🏆 TOP 3 COMPETITORS")
        for i, (company, row) in enumerate(competitors.head(3).iterrows(), 1):
            print(f"   {i}. {company}")
            print(f"      Revenue: £{row['revenue']:,.2f} | Events: {row['num_events']:,} | Market Share: {row['market_share_pct']:.1f}%")
        
        print(f"\n⚡ AXLE ENERGY BENCHMARK (Market Leader)")
        if axle:
            print(f"   Events: {axle['total_events']:,} ({axle['market_share_pct']:.1f}% market share)")
            print(f"   Avg Price: £{axle['avg_price']:.2f}/MWh")
            print(f"   Price Range: £{axle['price_range'][0]:.0f} - £{axle['price_range'][1]:.0f}/MWh")
            print(f"   Peak Hour: {axle['peak_hour']}:00")
            print(f"   Operating Zones: {axle['unique_zones']}")
        
        print(f"\n⏰ TEMPORAL INSIGHTS")
        peak_hour = hourly['num_events'].idxmax()
        print(f"   Peak Demand Hour: {peak_hour}:00 ({hourly.loc[peak_hour, 'num_events']} events)")
        peak_day = daily['num_events'].idxmax()
        print(f"   Peak Day: {peak_day} ({daily.loc[peak_day, 'num_events']} events)")
        
        print(f"\n💰 PRICING BENCHMARKS")
        print(f"   25th Percentile: £{price_stats['q25']:.2f}/MWh")
        print(f"   50th Percentile: £{price_stats['median']:.2f}/MWh")
        print(f"   75th Percentile: £{price_stats['q75']:.2f}/MWh")
        
        print("\n" + "="*70)
        
        return self.analysis_results
    
    def visualize_market(self, save_dir='outputs/visualizations'):
        """
        Generate all visualizations
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Market Share Pie Chart
        fig, ax = plt.subplots(figsize=(10, 8))
        competitors = self.analysis_results.get('competitors')
        if competitors is not None:
            top_companies = competitors.head(5)
            others_revenue = competitors.iloc[5:]['revenue'].sum() if len(competitors) > 5 else 0
            
            plot_data = list(top_companies['revenue']) + ([others_revenue] if others_revenue > 0 else [])
            plot_labels = list(top_companies.index) + (['Others'] if others_revenue > 0 else [])
            
            ax.pie(plot_data, labels=plot_labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Market Share by Company (Revenue)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/market_share.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Price Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.df['utilisation_price'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(self.df['utilisation_price'].median(), color='red', linestyle='--', 
                   label=f"Median: £{self.df['utilisation_price'].median():.0f}/MWh")
        ax.axvline(self.df['utilisation_price'].mean(), color='green', linestyle='--',
                   label=f"Mean: £{self.df['utilisation_price'].mean():.0f}/MWh")
        ax.set_xlabel('Utilisation Price (£/MWh)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Price Distribution - Day-Ahead Market', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Hourly Demand Pattern
        fig, ax = plt.subplots(figsize=(12, 6))
        temporal = self.analysis_results.get('temporal', {})
        if 'hourly' in temporal:
            hourly = temporal['hourly']
            ax.bar(hourly.index, hourly['num_events'], color='steelblue', edgecolor='black')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.set_title('Demand Pattern by Hour (Peak Window Identification)', fontsize=14, fontweight='bold')
            ax.set_xticks(range(0, 24))
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/hourly_demand.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Price vs Volume Scatter (Strategy Identification)
        fig, ax = plt.subplots(figsize=(12, 8))
        for company in self.df['company_name'].unique():
            company_data = self.df[self.df['company_name'] == company]
            avg_price = company_data['utilisation_price'].mean()
            total_events = len(company_data)
            ax.scatter(total_events, avg_price, s=200, alpha=0.6, label=company)
        
        ax.set_xlabel('Number of Events (Volume)', fontsize=12)
        ax.set_ylabel('Average Price (£/MWh)', fontsize=12)
        ax.set_title('Competitive Strategy Map: Price vs Volume', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/strategy_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Top Zones by Revenue
        fig, ax = plt.subplots(figsize=(12, 8))
        zones = self.analysis_results.get('zones')
        if zones is not None:
            top_zones = zones.head(10)
            ax.barh(range(len(top_zones)), top_zones['revenue'], color='coral', edgecolor='black')
            ax.set_yticks(range(len(top_zones)))
            ax.set_yticklabels(top_zones.index)
            ax.set_xlabel('Revenue (£)', fontsize=12)
            ax.set_title('Top 10 Zones by Revenue', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/top_zones.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ All visualizations saved to {save_dir}/")


# Convenience function for quick analysis
def analyze_ukpn_market(filepath='data/ukpnflexibilitydemandturndown.csv'):
    """
    Quick analysis function
    
    Returns:
    --------
    analyzer : UKPNMarketAnalyzer
        Analyzer object with all results
    """
    analyzer = UKPNMarketAnalyzer(filepath)
    analyzer.generate_comprehensive_report()
    analyzer.visualize_market()
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = analyze_ukpn_market('/mnt/project/ukpnflexibilitydemandturndown.csv')