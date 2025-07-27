#!/usr/bin/env python3
"""
NBA Archetype Radar Chart Generator
This script generates radar charts for all NBA player archetypes and saves them as images.
"""

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def load_and_prepare_data():
    """Load and prepare the NBA data for visualization."""
    print("Loading NBA data...")
    
    # Load the data
    df = pd.read_csv("nba_data_processed.csv")
    
    # Basic cleaning
    df = df.dropna(subset=["Player"])
    df = df[df["G"] >= 10]
    df = df[df["MP"] >= 10]
    
    # Fill missing values
    df["FG%"] = df["FG%"].fillna(0.0)
    df["3P%"] = df["3P%"].fillna(0.0)
    df["2P%"] = df["2P%"].fillna(0.0)
    df["FT%"] = df["FT%"].fillna(0.0)
    df["eFG%"] = df["eFG%"].fillna(0.0)
    
    # Calculate per-36-minute statistics
    stats_to_convert = ['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    for stat in stats_to_convert:
        column_name = f"{stat}_per36"
        df[column_name] = (df[stat] / df['MP']) * 36
    
    # Remove original columns
    original_columns_to_remove = ['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    df = df.drop(columns=original_columns_to_remove)
    
    return df

def perform_clustering(df):
    """Perform K-means clustering to create archetypes."""
    print("Performing clustering analysis...")
    
    # Prepare features for clustering
    X = df[['FG_per36', 'FGA_per36', '3P_per36', '3PA_per36', '2P_per36', '2PA_per36', 'FT_per36', 'FTA_per36', 'ORB_per36', 'DRB_per36', 'TRB_per36', 'AST_per36', 'STL_per36', 'BLK_per36', 'TOV_per36', 'PF_per36', 'PTS_per36']].copy()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=13, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled_df)
    
    # Add cluster labels to dataframe
    df['Cluster_K13'] = cluster_labels
    
    # Define archetype mapping
    archetype_mapping = {
        0: "Off-ball 3&D three-point shooters",
        1: "Scoring mobile centers",
        2: "Efficient mid-size scorers",
        3: "Veteran pure guards",
        4: "Mid-size defensive forwards",
        5: "High-volume perimeter playmakers",
        6: "Midcourt 3&D reserves",
        7: "Low-scoring, playmaking defensive wings",
        8: "Efficient rebounding paint protectors",
        9: "All-around high-usage bigs",
        10: "High-volume, versatile scorers",
        11: "Low-volume defensive playmakers",
        12: "High-impact playmaking scorers"
    }
    
    # Create archetype names column
    df['Archetype_Name'] = df['Cluster_K13'].map(archetype_mapping)
    
    return df

def generate_radar_charts(df):
    """Generate radar charts for all archetypes."""
    print("Generating radar charts...")
    
    # Create output directory
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define features for clustering
    features_for_clustering = [
        'PTS_per36', 'AST_per36', 'TRB_per36', '3P_per36', 'BLK_per36',
        'STL_per36', 'TOV_per36', 'eFG%'
    ]
    
    # Prepare cluster means for plotting
    cluster_means_for_plotting = df.groupby('Archetype_Name')[features_for_clustering].mean()
    
    radar_features = [
        'PTS_per36', 'AST_per36', 'TRB_per36', '3P_per36', 'BLK_per36',
        'STL_per36', 'TOV_per36', 'eFG%'
    ]
    
    # Scale data for radar charts
    scaler_radar = MinMaxScaler()
    scaled_radar_data = scaler_radar.fit_transform(cluster_means_for_plotting[radar_features])
    scaled_radar_df = pd.DataFrame(scaled_radar_data, columns=radar_features, index=cluster_means_for_plotting.index)
    
    # Transform to long format for Plotly Express
    df_long_format = scaled_radar_df.reset_index().melt(
        id_vars=['Archetype_Name'],
        value_vars=radar_features,
        var_name='Statistic',
        value_name='Scaled_Value'
    )
    
    # Generate individual radar charts for each archetype
    print("Creating individual archetype radar charts...")
    archetype_files = {}
    
    for archetype_name in df_long_format['Archetype_Name'].unique():
        df_single_archetype = df_long_format[df_long_format['Archetype_Name'] == archetype_name]
        
        fig = px.line_polar(
            df_single_archetype,
            r='Scaled_Value',
            theta='Statistic',
            line_close=True,
            title=f'Archetype Profile: {archetype_name}',
            labels={'Scaled_Value': 'Scaled Value (0-1)', 'Statistic': 'Statistic'},
            height=650,
            width=750
        )
        
        fig.update_traces(fill='toself')
        
        # Customize layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=['0', '0.25', '0.5', '0.75', '1'],
                    gridcolor='lightgray',
                    linecolor='lightgray'
                ),
                angularaxis=dict(
                    rotation=90,
                    direction="clockwise",
                    tickfont=dict(size=10),
                    linecolor='gray',
                    linewidth=0.5,
                    gridcolor='lightgray'
                )
            ),
            title_font_size=16,
            title_x=0.5
        )
        
        # Create filename based on archetype
        if archetype_name == "Off-ball 3&D three-point shooters":
            filename = "archetype_0_off_ball_3d_shooter_radar.png"
        elif archetype_name == "Scoring mobile centers":
            filename = "archetype_1_scoring_mobile_center_radar.png"
        elif archetype_name == "Efficient mid-size scorers":
            filename = "archetype_2_efficient_mid_size_scorer_radar.png"
        elif archetype_name == "Veteran pure guards":
            filename = "archetype_3_veteran_pure_guard_radar.png"
        elif archetype_name == "Mid-size defensive forwards":
            filename = "archetype_4_mid_size_defensive_forward_radar.png"
        elif archetype_name == "High-volume perimeter playmakers":
            filename = "archetype_5_high_volume_playmaker_radar.png"
        elif archetype_name == "Midcourt 3&D reserves":
            filename = "archetype_6_midcourt_3d_reserve_radar.png"
        elif archetype_name == "Low-scoring, playmaking defensive wings":
            filename = "archetype_7_low_scoring_defensive_wing_radar.png"
        elif archetype_name == "Efficient rebounding paint protectors":
            filename = "archetype_8_efficient_paint_protector_radar.png"
        elif archetype_name == "All-around high-usage bigs":
            filename = "archetype_9_all_around_high_usage_big_radar.png"
        elif archetype_name == "High-volume, versatile scorers":
            filename = "archetype_10_all_around_offensive_shooter_radar.png"
        elif archetype_name == "Low-volume defensive playmakers":
            filename = "archetype_11_low_volume_defensive_playmaker_radar.png"
        elif archetype_name == "High-impact playmaking scorers":
            filename = "archetype_12_high_impact_playmaking_scorer_radar.png"
        else:
            # Fallback for any other names
            filename_safe = archetype_name.replace(' ', '_').replace('/', '_').replace('-', '_').lower()
            filename = f"archetype_{filename_safe}_radar.png"
        
        output_path = os.path.join(output_dir, filename)
        fig.write_image(output_path)
        print(f"Saved: {output_path}")
        archetype_files[archetype_name] = filename
    
    # Generate combined radar chart
    print("Creating combined radar chart...")
    fig_all = px.line_polar(
        df_long_format,
        r='Scaled_Value',
        theta='Statistic',
        line_close=True,
        color='Archetype_Name',
        title='All Player Archetypes Comparison (Scaled Features)',
        labels={'Scaled_Value': 'Scaled Value (0-1)', 'Statistic': 'Statistic'},
        height=600,
        width=800
    )
    fig_all.update_traces(fill='toself')
    
    # Save combined chart
    combined_output_path = os.path.join(output_dir, "all_archetypes_comparison_radar.png")
    fig_all.write_image(combined_output_path)
    print(f"Saved: {combined_output_path}")
    
    return output_dir, archetype_files

def main():
    """Main function to run the radar chart generation."""
    print("=== NBA Archetype Radar Chart Generator ===")
    print()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Perform clustering
        df = perform_clustering(df)
        
        # Generate radar charts
        output_dir, archetype_files = generate_radar_charts(df)
        
        print()
        print("=== Generation Complete ===")
        print(f"All radar charts have been saved to the '{output_dir}' directory.")
        print()
        print("Files generated:")
        print("- all_archetypes_comparison_radar.png (combined view of all archetypes)")
        for archetype_name, filename in archetype_files.items():
            print(f"- {filename} ({archetype_name})")
        print()
        print("You can now include these images in your markdown files using:")
        print("![Description](images/filename.png)")
        
    except FileNotFoundError:
        print("Error: Could not find 'nba_data_processed.csv' file.")
        print("Please make sure the data file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main() 