"""
Grid Analysis and Visualization Tools for VIC Surrogate Model
Author: Siddik Barbhuiya
Description: Tools to analyze and visualize grid locations used in training/validation/testing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path


class GridAnalyzer:
    """
    Analyze and visualize grid locations used in the surrogate model
    Author: Siddik Barbhuiya
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize grid analyzer
        
        Args:
            output_dir: Directory containing grid location files
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load grid data for all modes
        self.grid_data = {}
        self._load_all_grid_data()
    
    def _load_all_grid_data(self):
        """Load grid data for all available modes"""
        modes = ['train', 'valid', 'test']
        
        for mode in modes:
            csv_file = self.output_dir / f'grid_locations_{mode}.csv'
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    self.grid_data[mode] = df
                    self.logger.info(f"Loaded {len(df)} {mode} grid locations")
                except Exception as e:
                    self.logger.warning(f"Could not load {mode} grid data: {e}")
    
    def create_comprehensive_report(self) -> str:
        """
        Create a comprehensive analysis report of grid locations
        
        Returns:
            String containing the full report
        """
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE GRID LOCATION ANALYSIS REPORT",
            "Author: Siddik Barbhuiya",
            "=" * 80,
            ""
        ]
        
        # Overview
        report_lines.extend(self._get_overview())
        
        # Spatial distribution analysis
        report_lines.extend(self._get_spatial_analysis())
        
        # Grid overlap analysis
        report_lines.extend(self._get_overlap_analysis())
        
        # Grid index analysis
        report_lines.extend(self._get_index_analysis())
        
        # Statistics summary
        report_lines.extend(self._get_statistics_summary())
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _get_overview(self) -> List[str]:
        """Get overview of grid data"""
        lines = ["OVERVIEW:", "-" * 20, ""]
        
        for mode, df in self.grid_data.items():
            lines.append(f"{mode.upper()} MODE:")
            lines.append(f"  Total grids: {len(df)}")
            lines.append(f"  Latitude range: {df['latitude'].min():.3f} to {df['latitude'].max():.3f}")
            lines.append(f"  Longitude range: {df['longitude'].min():.3f} to {df['longitude'].max():.3f}")
            if 'scenario_start' in df.columns:
                lines.append(f"  Scenario range: {df['scenario_start'].iloc[0]} to {df['scenario_end'].iloc[0]}")
            lines.append("")
        
        return lines
    
    def _get_spatial_analysis(self) -> List[str]:
        """Analyze spatial distribution of grids"""
        lines = ["SPATIAL DISTRIBUTION ANALYSIS:", "-" * 35, ""]
        
        if len(self.grid_data) == 0:
            lines.append("No grid data available for analysis.")
            return lines
        
        # Combine all grids for spatial analysis
        all_grids = []
        for mode, df in self.grid_data.items():
            df_copy = df.copy()
            df_copy['mode'] = mode
            all_grids.append(df_copy)
        
        if all_grids:
            combined_df = pd.concat(all_grids, ignore_index=True)
            
            # Spatial statistics
            lines.append("Combined Spatial Statistics:")
            lines.append(f"  Total unique grid locations: {len(combined_df.drop_duplicates(['latitude', 'longitude']))}")
            lines.append(f"  Overall latitude range: {combined_df['latitude'].min():.3f} to {combined_df['latitude'].max():.3f}")
            lines.append(f"  Overall longitude range: {combined_df['longitude'].min():.3f} to {combined_df['longitude'].max():.3f}")
            lines.append(f"  Spatial extent: {combined_df['latitude'].max() - combined_df['latitude'].min():.3f}° lat × {combined_df['longitude'].max() - combined_df['longitude'].min():.3f}° lon")
            lines.append("")
            
            # Mode-specific spatial characteristics
            for mode, df in self.grid_data.items():
                centroid_lat = df['latitude'].mean()
                centroid_lon = df['longitude'].mean()
                lines.append(f"{mode.upper()} spatial characteristics:")
                lines.append(f"  Centroid: ({centroid_lat:.3f}°N, {centroid_lon:.3f}°E)")
                lines.append(f"  Latitude std: {df['latitude'].std():.3f}°")
                lines.append(f"  Longitude std: {df['longitude'].std():.3f}°")
                lines.append("")
        
        return lines
    
    def _get_overlap_analysis(self) -> List[str]:
        """Analyze grid overlap between modes"""
        lines = ["GRID OVERLAP ANALYSIS:", "-" * 25, ""]
        
        if len(self.grid_data) < 2:
            lines.append("Need at least 2 modes for overlap analysis.")
            return lines
        
        modes = list(self.grid_data.keys())
        
        # Check overlaps between all pairs
        for i, mode1 in enumerate(modes):
            for mode2 in modes[i+1:]:
                df1 = self.grid_data[mode1]
                df2 = self.grid_data[mode2]
                
                # Find overlapping grid locations
                merged = pd.merge(
                    df1[['latitude', 'longitude', 'global_flat_index']],
                    df2[['latitude', 'longitude', 'global_flat_index']],
                    on=['latitude', 'longitude'],
                    how='inner'
                )
                
                overlap_count = len(merged)
                overlap_pct1 = 100 * overlap_count / len(df1)
                overlap_pct2 = 100 * overlap_count / len(df2)
                
                lines.append(f"{mode1.upper()} vs {mode2.upper()}:")
                lines.append(f"  Overlapping grids: {overlap_count}")
                lines.append(f"  Overlap as % of {mode1}: {overlap_pct1:.1f}%")
                lines.append(f"  Overlap as % of {mode2}: {overlap_pct2:.1f}%")
                
                if overlap_count > 0:
                    lines.append(f"  Example overlapping locations:")
                    for idx in range(min(3, len(merged))):
                        lat = merged.iloc[idx]['latitude']
                        lon = merged.iloc[idx]['longitude']
                        lines.append(f"    Grid {idx+1}: ({lat:.3f}°N, {lon:.3f}°E)")
                lines.append("")
        
        return lines
    
    def _get_index_analysis(self) -> List[str]:
        """Analyze grid indices"""
        lines = ["GRID INDEX ANALYSIS:", "-" * 22, ""]
        
        for mode, df in self.grid_data.items():
            lines.append(f"{mode.upper()} grid indices:")
            lines.append(f"  Global flat index range: {df['global_flat_index'].min()} to {df['global_flat_index'].max()}")
            lines.append(f"  I-index range: {df['i_index'].min()} to {df['i_index'].max()}")
            lines.append(f"  J-index range: {df['j_index'].min()} to {df['j_index'].max()}")
            lines.append(f"  Unique global indices: {df['global_flat_index'].nunique()}")
            lines.append("")
        
        return lines
    
    def _get_statistics_summary(self) -> List[str]:
        """Get statistical summary"""
        lines = ["STATISTICAL SUMMARY:", "-" * 20, ""]
        
        for mode, df in self.grid_data.items():
            lines.append(f"{mode.upper()} statistics:")
            
            # Latitude statistics
            lat_stats = df['latitude'].describe()
            lines.append(f"  Latitude: mean={lat_stats['mean']:.3f}°, std={lat_stats['std']:.3f}°")
            
            # Longitude statistics  
            lon_stats = df['longitude'].describe()
            lines.append(f"  Longitude: mean={lon_stats['mean']:.3f}°, std={lon_stats['std']:.3f}°")
            
            lines.append("")
        
        return lines
    
    def create_spatial_plots(self, save_dir: Optional[str] = None):
        """
        Create comprehensive spatial visualization plots
        
        Args:
            save_dir: Directory to save plots (defaults to output_dir)
        """
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
            
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: All modes on single map
        self._plot_all_modes_map(save_dir)
        
        # Plot 2: Individual mode maps
        self._plot_individual_mode_maps(save_dir)
        
        # Plot 3: Spatial distribution histograms
        self._plot_spatial_histograms(save_dir)
        
        # Plot 4: Grid index scatter plots
        self._plot_grid_indices(save_dir)
        
        self.logger.info(f"Spatial plots saved to {save_dir}")
    
    def _plot_all_modes_map(self, save_dir: Path):
        """Plot all modes on a single map"""
        plt.figure(figsize=(12, 8))
        
        colors = {'train': 'blue', 'valid': 'green', 'test': 'red'}
        markers = {'train': 'o', 'valid': 's', 'test': '^'}
        
        for mode, df in self.grid_data.items():
            plt.scatter(
                df['longitude'], df['latitude'],
                c=colors.get(mode, 'black'), 
                marker=markers.get(mode, 'o'),
                alpha=0.6, s=20,
                label=f'{mode.title()} ({len(df)} grids)'
            )
        
        plt.xlabel('Longitude (°E)')
        plt.ylabel('Latitude (°N)')
        plt.title('Grid Locations for All Modes\nBy Siddik')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'all_modes_grid_locations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_mode_maps(self, save_dir: Path):
        """Plot individual maps for each mode"""
        for mode, df in self.grid_data.items():
            plt.figure(figsize=(10, 6))
            
            scatter = plt.scatter(
                df['longitude'], df['latitude'],
                c=df['dataset_grid_index'], 
                cmap='viridis',
                alpha=0.7, s=30
            )
            
            plt.colorbar(scatter, label='Dataset Grid Index')
            plt.xlabel('Longitude (°E)')
            plt.ylabel('Latitude (°N)')
            plt.title(f'{mode.title()} Mode Grid Locations ({len(df)} grids)\nBy Siddik')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(save_dir / f'{mode}_grid_locations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_spatial_histograms(self, save_dir: Path):
        """Plot spatial distribution histograms"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spatial Distribution Histograms\nBy Siddik', fontsize=14)
        
        # Latitude histogram
        for mode, df in self.grid_data.items():
            axes[0, 0].hist(df['latitude'], alpha=0.6, label=mode, bins=20)
        axes[0, 0].set_xlabel('Latitude (°N)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Latitude Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Longitude histogram
        for mode, df in self.grid_data.items():
            axes[0, 1].hist(df['longitude'], alpha=0.6, label=mode, bins=20)
        axes[0, 1].set_xlabel('Longitude (°E)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Longitude Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D histogram for first mode (if available)
        if self.grid_data:
            mode = list(self.grid_data.keys())[0]
            df = self.grid_data[mode]
            axes[1, 0].hist2d(df['longitude'], df['latitude'], bins=15, cmap='Blues')
            axes[1, 0].set_xlabel('Longitude (°E)')
            axes[1, 0].set_ylabel('Latitude (°N)')
            axes[1, 0].set_title(f'2D Distribution ({mode.title()})')
        
        # Grid count comparison
        if len(self.grid_data) > 1:
            modes = list(self.grid_data.keys())
            counts = [len(self.grid_data[mode]) for mode in modes]
            axes[1, 1].bar(modes, counts, alpha=0.7)
            axes[1, 1].set_ylabel('Number of Grids')
            axes[1, 1].set_title('Grid Count by Mode')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'spatial_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_grid_indices(self, save_dir: Path):
        """Plot grid indices analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Grid Index Analysis\nBy Siddik', fontsize=14)
        
        for mode, df in self.grid_data.items():
            # I vs J indices
            axes[0].scatter(df['j_index'], df['i_index'], alpha=0.6, label=mode, s=20)
            
            # Global flat index distribution
            axes[1].hist(df['global_flat_index'], alpha=0.6, label=mode, bins=20)
            
            # Dataset vs global index
            axes[2].scatter(df['dataset_grid_index'], df['global_flat_index'], 
                          alpha=0.6, label=mode, s=20)
        
        axes[0].set_xlabel('J Index (Column)')
        axes[0].set_ylabel('I Index (Row)')
        axes[0].set_title('Spatial Array Indices')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Global Flat Index')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Global Index Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Dataset Grid Index')
        axes[2].set_ylabel('Global Flat Index')
        axes[2].set_title('Index Mapping')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'grid_indices_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_grid_summary_table(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export a summary table of grid information
        
        Args:
            output_file: Optional file path to save the summary table
            
        Returns:
            DataFrame containing grid summary
        """
        summary_data = []
        
        for mode, df in self.grid_data.items():
            summary = {
                'mode': mode,
                'total_grids': len(df),
                'lat_min': df['latitude'].min(),
                'lat_max': df['latitude'].max(),
                'lat_mean': df['latitude'].mean(),
                'lat_std': df['latitude'].std(),
                'lon_min': df['longitude'].min(),
                'lon_max': df['longitude'].max(), 
                'lon_mean': df['longitude'].mean(),
                'lon_std': df['longitude'].std(),
                'spatial_extent_lat': df['latitude'].max() - df['latitude'].min(),
                'spatial_extent_lon': df['longitude'].max() - df['longitude'].min(),
                'unique_flat_indices': df['global_flat_index'].nunique()
            }
            
            if 'scenario_start' in df.columns:
                summary['scenario_start'] = df['scenario_start'].iloc[0]
                summary['scenario_end'] = df['scenario_end'].iloc[0]
                summary['scenario_count'] = df['scenario_end'].iloc[0] - df['scenario_start'].iloc[0] + 1
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        if output_file:
            summary_df.to_csv(output_file, index=False)
            self.logger.info(f"Grid summary table saved to {output_file}")
        
        return summary_df


def analyze_grid_locations(output_dir: str, create_plots: bool = True) -> str:
    """
    Convenience function to analyze grid locations
    
    Args:
        output_dir: Directory containing grid location files
        create_plots: Whether to create visualization plots
        
    Returns:
        Analysis report as string
    """
    analyzer = GridAnalyzer(output_dir)
    
    # Create comprehensive report
    report = analyzer.create_comprehensive_report()
    
    # Save report to file
    report_file = Path(output_dir) / 'grid_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create plots if requested
    if create_plots:
        analyzer.create_spatial_plots()
    
    # Create summary table
    summary_df = analyzer.export_grid_summary_table(
        Path(output_dir) / 'grid_summary_table.csv'
    )
    
    print(f"Grid analysis complete. Files saved to {output_dir}")
    return report