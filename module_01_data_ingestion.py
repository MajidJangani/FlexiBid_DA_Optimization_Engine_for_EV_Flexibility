# module_01_data_ingestion.py
"""
Data Ingestion Module
=====================
Fetches and prepares all input data aligned with UKPN timeline:
1. Octopus Energy tariff data (ToU pricing) - May 2024 to Dec 2025
2. Vehicle specifications
3. Fleet behavior data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os


class OctopusTariffFetcher:
    """
    Fetches Octopus Agile tariff data aligned with UKPN event timeline
    """
    
    def __init__(self, region='A'):
        """
        Initialize fetcher
        
        Parameters:
        -----------
        region : str
            Tariff region (A = Eastern England - UKPN area)
        """
        self.region = region
        self.product_code = "AGILE-FLEX-22-11-25"
        self.tariff_code = f"E-1R-{self.product_code}-{region}"
        self.base_url = f"https://api.octopus.energy/v1/products/{self.product_code}/electricity-tariffs/{self.tariff_code}/standard-unit-rates/"
        
    def extract_ukpn_date_range(self, ukpn_filepath):
        """
        Extract start and end dates from UKPN data
        
        Returns:
        --------
        tuple: (start_date, end_date)
        """
        df = pd.read_csv(ukpn_filepath)
        df['start_time_utc'] = pd.to_datetime(df['start_time_utc'], utc=True)
        
        start_date = df['start_time_utc'].min()
        end_date = df['start_time_utc'].max()
        
        print(f"📅 UKPN Date Range:")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}")
        print(f"   Duration: {(end_date - start_date).days} days")
        
        return start_date, end_date
    
    def fetch_tariff_chunk(self, period_from, period_to, retry_count=3):
        """
        Fetch tariff data for a specific date range
        
        Parameters:
        -----------
        period_from : datetime
        period_to : datetime
        retry_count : int
            Number of retries on failure
        
        Returns:
        --------
        pd.DataFrame or None
        """
        params = {
            'period_from': period_from.isoformat(),
            'period_to': period_to.isoformat(),
            'page_size': 1500  # Max per request
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    df = pd.DataFrame(data['results'])
                    df['valid_from'] = pd.to_datetime(df['valid_from'], utc=True)
                    df['valid_to'] = pd.to_datetime(df['valid_to'], utc=True)
                    return df
                else:
                    print(f"   ⚠️  No data returned for {period_from.date()} to {period_to.date()}")
                    return None
                
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️  Attempt {attempt + 1}/{retry_count} failed: {str(e)[:100]}")
                if attempt < retry_count - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    return None
    
    def fetch_full_timeline(self, start_date, end_date, chunk_days=30):
        """
        Fetch tariff data for entire UKPN timeline in chunks
        
        Parameters:
        -----------
        start_date : datetime
        end_date : datetime
        chunk_days : int
            Size of each fetch chunk (API limits apply)
        
        Returns:
        --------
        pd.DataFrame with all fetched tariff data
        """
        print(f"\n🔌 Fetching Octopus Agile tariff data...")
        print(f"   Region: {self.region} (Eastern England - UKPN)")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Strategy: Fetch in {chunk_days}-day chunks\n")
        
        all_data = []
        current_start = start_date
        chunk_num = 0
        
        while current_start < end_date:
            chunk_num += 1
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            print(f"   Chunk {chunk_num}: {current_start.date()} to {current_end.date()}...", end=" ")
            
            chunk_data = self.fetch_tariff_chunk(current_start, current_end)
            
            if chunk_data is not None:
                all_data.append(chunk_data)
                print(f"✅ {len(chunk_data)} records")
            else:
                print(f"❌ Failed")
            
            current_start = current_end
            time.sleep(0.5)  # Rate limiting
        
        if not all_data:
            print("\n❌ No tariff data fetched successfully!")
            return None
        
        # Combine all chunks
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('valid_from').reset_index(drop=True)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['valid_from'], keep='first')
        
        print(f"\n✅ Fetched {len(combined)} tariff periods")
        print(f"   Date range: {combined['valid_from'].min()} to {combined['valid_to'].max()}")
        print(f"   Price range: {combined['value_inc_vat'].min():.2f}p - {combined['value_inc_vat'].max():.2f}p/kWh")
        
        return combined
    
    def identify_gaps(self, tariff_df, start_date, end_date):
        """
        Identify missing 30-minute periods in fetched data
        
        Returns:
        --------
        list of tuples: [(gap_start, gap_end), ...]
        """
        # Create complete 30-min timeline
        expected_periods = pd.date_range(
            start=start_date,
            end=end_date,
            freq='30min',
            tz='UTC'
        )
        
        # Find actual periods
        actual_periods = set(tariff_df['valid_from'])
        
        # Find gaps
        gaps = []
        gap_start = None
        
        for period in expected_periods:
            if period not in actual_periods:
                if gap_start is None:
                    gap_start = period
            else:
                if gap_start is not None:
                    gaps.append((gap_start, period))
                    gap_start = None
        
        # Close final gap if exists
        if gap_start is not None:
            gaps.append((gap_start, expected_periods[-1]))
        
        if gaps:
            print(f"\n⚠️  Found {len(gaps)} gaps in tariff data:")
            total_missing = sum([(gap[1] - gap[0]).total_seconds() / 1800 for gap in gaps])
            print(f"   Total missing periods: {int(total_missing)}")
            for i, (gap_start, gap_end) in enumerate(gaps[:5], 1):  # Show first 5
                print(f"   {i}. {gap_start.date()} to {gap_end.date()}")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps) - 5} more gaps")
        else:
            print(f"\n✅ No gaps detected - complete timeline!")
        
        return gaps
    
    def fill_gaps(self, tariff_df, gaps, method='seasonal'):
        """
        Fill missing tariff data
        
        Parameters:
        -----------
        tariff_df : pd.DataFrame
            Existing tariff data
        gaps : list
            List of (gap_start, gap_end) tuples
        method : str
            'seasonal' = use hour-of-day averages from same season
            'interpolate' = linear interpolation
            'forward_fill' = carry forward last known value
        
        Returns:
        --------
        pd.DataFrame with gaps filled
        """
        if not gaps:
            return tariff_df
        
        print(f"\n🔧 Filling gaps using '{method}' method...")
        
        filled_df = tariff_df.copy()
        filled_records = []
        
        if method == 'seasonal':
            # Calculate seasonal hour-of-day averages
            filled_df['hour'] = filled_df['valid_from'].dt.hour
            filled_df['month'] = filled_df['valid_from'].dt.month
            filled_df['season'] = filled_df['month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            })
            
            # Average price by season and hour
            seasonal_avg = filled_df.groupby(['season', 'hour'])['value_inc_vat'].mean()
            
            for gap_start, gap_end in gaps:
                gap_periods = pd.date_range(gap_start, gap_end, freq='30min', tz='UTC')
                
                for period in gap_periods:
                    season = {12: 'winter', 1: 'winter', 2: 'winter',
                             3: 'spring', 4: 'spring', 5: 'spring',
                             6: 'summer', 7: 'summer', 8: 'summer',
                             9: 'autumn', 10: 'autumn', 11: 'autumn'}[period.month]
                    hour = period.hour
                    
                    # Get seasonal average for this hour
                    if (season, hour) in seasonal_avg.index:
                        price = seasonal_avg.loc[(season, hour)]
                    else:
                        # Fallback to overall hour average
                        price = filled_df[filled_df['hour'] == hour]['value_inc_vat'].mean()
                        if pd.isna(price):
                            price = filled_df['value_inc_vat'].mean()  # Last resort
                    
                    filled_records.append({
                        'valid_from': period,
                        'valid_to': period + timedelta(minutes=30),
                        'value_inc_vat': price,
                        'is_filled': True  # Mark as filled data
                    })
        
        elif method == 'forward_fill':
            for gap_start, gap_end in gaps:
                # Find last known value before gap
                prev_data = filled_df[filled_df['valid_from'] < gap_start].sort_values('valid_from')
                if len(prev_data) > 0:
                    last_price = prev_data.iloc[-1]['value_inc_vat']
                else:
                    last_price = filled_df['value_inc_vat'].mean()
                
                gap_periods = pd.date_range(gap_start, gap_end, freq='30min', tz='UTC')
                
                for period in gap_periods:
                    filled_records.append({
                        'valid_from': period,
                        'valid_to': period + timedelta(minutes=30),
                        'value_inc_vat': last_price,
                        'is_filled': True
                    })
        
        # Combine original and filled data
        if filled_records:
            filled_df_new = pd.DataFrame(filled_records)
            filled_df['is_filled'] = False
            result = pd.concat([filled_df, filled_df_new], ignore_index=True)
            result = result.sort_values('valid_from').reset_index(drop=True)
            
            filled_count = result['is_filled'].sum()
            print(f"✅ Filled {filled_count} missing periods ({filled_count/len(result)*100:.1f}% of total)")
            
            return result
        
        return filled_df
    
    def save_tariff_data(self, tariff_df, filepath='data/octopus_tariffs.csv'):
        """Save tariff data with metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save main data
        tariff_df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'fetch_date': datetime.now().isoformat(),
            'region': self.region,
            'product_code': self.product_code,
            'total_periods': len(tariff_df),
            'filled_periods': int(tariff_df['is_filled'].sum()) if 'is_filled' in tariff_df.columns else 0,
            'date_range': {
                'start': tariff_df['valid_from'].min().isoformat(),
                'end': tariff_df['valid_to'].max().isoformat()
            },
            'price_stats': {
                'min': float(tariff_df['value_inc_vat'].min()),
                'max': float(tariff_df['value_inc_vat'].max()),
                'mean': float(tariff_df['value_inc_vat'].mean()),
                'median': float(tariff_df['value_inc_vat'].median())
            }
        }
        
        metadata_path = filepath.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Tariff data saved to {filepath}")
        print(f"✅ Metadata saved to {metadata_path}")


def fetch_and_prepare_tariff_data(ukpn_filepath, output_path='data/octopus_tariffs.csv', 
                                   region='A', fill_method='seasonal'):
    """
    Main function to fetch and prepare tariff data aligned with UKPN timeline
    
    Parameters:
    -----------
    ukpn_filepath : str
        Path to UKPN CSV
    output_path : str
        Where to save tariff data
    region : str
        Octopus tariff region
    fill_method : str
        Method to fill gaps
    
    Returns:
    --------
    pd.DataFrame: Complete tariff data
    """
    fetcher = OctopusTariffFetcher(region=region)
    
    # Step 1: Extract UKPN date range
    start_date, end_date = fetcher.extract_ukpn_date_range(ukpn_filepath)
    
    # Step 2: Fetch tariff data
    tariff_data = fetcher.fetch_full_timeline(start_date, end_date)
    
    if tariff_data is None:
        print("\n❌ Failed to fetch tariff data. Exiting.")
        return None
    
    # Step 3: Identify gaps
    gaps = fetcher.identify_gaps(tariff_data, start_date, end_date)
    
    # Step 4: Fill gaps
    complete_tariff = fetcher.fill_gaps(tariff_data, gaps, method=fill_method)
    
    # Step 5: Save
    fetcher.save_tariff_data(complete_tariff, filepath=output_path)
    
    return complete_tariff


if __name__ == "__main__":
    # Test the fetcher
    tariff_data = fetch_and_prepare_tariff_data(
        ukpn_filepath='/mnt/project/ukpnflexibilitydemandturndown.csv',
        output_path='data/octopus_tariffs.csv',
        region='A',  # Eastern England
        fill_method='seasonal'
    )
    
    if tariff_data is not None:
        print(f"\n✅ SUCCESS: {len(tariff_data)} tariff periods ready for use")