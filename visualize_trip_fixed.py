import pickle
import numpy as np
import pandas as pd
import folium
import random
import os
from pathlib import Path

def load_trajectory_data(file_path):
    """Load trajectory data from pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_detailed_trajectory(drives_obj, drive_id):
    """Extract detailed GPS trajectory for a specific drive"""
    if hasattr(drives_obj, 'all_drives') and drives_obj.all_drives:
        # Look for the specific drive_id in all_drives
        for i, trip in enumerate(drives_obj.all_drives):
            if isinstance(trip, pd.DataFrame):
                if 'drive_id' in trip.columns:
                    if drive_id in trip['drive_id'].values:
                        print(f"Found exact match for drive {drive_id} in all_drives[{i}]")
                        return trip
                
                # If no drive_id column, check if this trip has reasonable GPS data
                if 'latitude' in trip.columns and 'longitude' in trip.columns:
                    # Check if coordinates look like real GPS coordinates (not normalized)
                    lat_vals = trip['latitude'].dropna()
                    lon_vals = trip['longitude'].dropna()
                    
                    if len(lat_vals) > 0 and len(lon_vals) > 0:
                        lat_mean = lat_vals.mean()
                        lon_mean = lon_vals.mean()
                        
                        # Check if these look like real coordinates (not normalized 0-1)
                        if (30 <= lat_mean <= 35 and 34 <= lon_mean <= 36):  # Israel coordinate range
                            print(f"Found trip with real coordinates in all_drives[{i}]")
                            return trip
    return None

def create_speed_segments(detailed_trajectory):
    """Create truly disconnected segments - only first/last speed=0 points per segment"""
    if 'speed' not in detailed_trajectory.columns:
        print("No speed data available for segmentation")
        return None
    
    # Get coordinates and speed data
    lats = detailed_trajectory['latitude'].values
    lons = detailed_trajectory['longitude'].values
    speeds = detailed_trajectory['speed'].values
    
    segments = []
    current_segment = []
    in_motion = False
    last_zero_point = None
    
    for i, (lat, lon, speed) in enumerate(zip(lats, lons, speeds)):
        if pd.isna(lat) or pd.isna(lon) or pd.isna(speed):
            continue
            
        if speed == 0:
            # Store the zero-speed point, but don't add to segment yet
            last_zero_point = (lat, lon, speed, i)
            
            # If we were in motion and now stopped, end the segment
            if in_motion:
                current_segment.append(last_zero_point)  # Add the stopping point
                if len(current_segment) >= 2:  # Only add segments with at least 2 points
                    segments.append(current_segment.copy())
                current_segment = []
                in_motion = False
                
        else:  # speed > 0
            # If we're starting to move after being stopped
            if not in_motion:
                current_segment = []
                # Add the last zero point as segment start (if we have one)
                if last_zero_point is not None:
                    current_segment.append(last_zero_point)
                current_segment.append((lat, lon, speed, i))
                in_motion = True
                last_zero_point = None
            else:
                # Continue adding moving points
                current_segment.append((lat, lon, speed, i))
    
    # Add the last segment if vehicle was still in motion at the end
    if in_motion and len(current_segment) >= 2:
        segments.append(current_segment)
    
    print(f"Created {len(segments)} clean segments (only first/last speed=0 points)")
    return segments

def visualize_random_trip():
    """Visualize a random trip with both complete and segmented views"""
    
    # Get list of data files
    data_dir = Path("data/raw")
    pkl_files = list(data_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("No pickle files found in data/raw directory")
        return
    
    # Select a random file
    random_file = random.choice(pkl_files)
    print(f"Loading data from: {random_file}")
    
    try:
        drives_obj = load_trajectory_data(random_file)
        print(f"Data type: {type(drives_obj)}")
        print(f"Car ID: {getattr(drives_obj, 'car_id', 'Unknown')}")
        
        # Get metadata with actual coordinates
        if hasattr(drives_obj, 'df') and drives_obj.df is not None:
            df = drives_obj.df
            print(f"Available drives: {len(df)}")
            
            # Select a random drive
            random_idx = random.randint(0, len(df) - 1)
            drive_row = df.iloc[random_idx]
            drive_id = drive_row.name
            
            print(f"Selected drive {random_idx}: {drive_id}")
            
            # Extract start and end coordinates from metadata
            start_lat = drive_row['START_LATITUDE']
            start_lon = drive_row['START_LONGITUDE']
            end_lat = drive_row['END_LATITUDE']
            end_lon = drive_row['END_LONGITUDE']
            
            print(f"Start coordinates: ({start_lat:.6f}, {start_lon:.6f})")
            print(f"End coordinates: ({end_lat:.6f}, {end_lon:.6f})")
            
            # Try to get the detailed trajectory
            detailed_trajectory = extract_detailed_trajectory(drives_obj, drive_id)
            
            # If no detailed trajectory found, try any trip with real coordinates
            if detailed_trajectory is None and hasattr(drives_obj, 'all_drives'):
                print("No exact match found, looking for any trip with real GPS coordinates...")
                for i, trip in enumerate(drives_obj.all_drives):
                    if isinstance(trip, pd.DataFrame) and 'latitude' in trip.columns and 'longitude' in trip.columns:
                        lat_vals = trip['latitude'].dropna()
                        lon_vals = trip['longitude'].dropna()
                        
                        if len(lat_vals) > 10:  # Ensure reasonable number of points
                            lat_range = lat_vals.max() - lat_vals.min()
                            lon_range = lon_vals.max() - lon_vals.min()
                            
                            # Check if coordinates have reasonable variation (not all same point)
                            if lat_range > 0.001 and lon_range > 0.001:
                                # Check if coordinates are in reasonable range for real GPS
                                if (lat_vals.min() > 25 and lat_vals.max() < 40 and 
                                    lon_vals.min() > 30 and lon_vals.max() < 40):
                                    print(f"Using trip {i} with {len(lat_vals)} GPS points")
                                    detailed_trajectory = trip
                                    break
            
            # Create the maps
            if detailed_trajectory is not None:
                # Use detailed trajectory coordinates
                lats = detailed_trajectory['latitude'].dropna().values
                lons = detailed_trajectory['longitude'].dropna().values
                
                # Filter out any invalid coordinates
                valid_mask = (
                    (~np.isnan(lats)) & 
                    (~np.isnan(lons)) & 
                    (lats != 0) & 
                    (lons != 0) &
                    (np.abs(lats) <= 90) &
                    (np.abs(lons) <= 180)
                )
                
                lats = lats[valid_mask]
                lons = lons[valid_mask]
                
                if len(lats) == 0:
                    print("No valid GPS coordinates found in detailed trajectory")
                    return
                
                print(f"Found {len(lats)} detailed GPS points")
                print(f"Latitude range: {lats.min():.6f} to {lats.max():.6f}")
                print(f"Longitude range: {lons.min():.6f} to {lons.max():.6f}")
                
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
                
                # ============== MAP 1: COMPLETE TRAJECTORY ==============
                create_complete_trajectory_map(lats, lons, center_lat, center_lon, drive_id, drives_obj)
                
                # ============== MAP 2: SEGMENTED TRAJECTORY ==============
                create_segmented_trajectory_map(detailed_trajectory, center_lat, center_lon, drive_id, drives_obj)
                
            else:
                print("No detailed trajectory found - cannot create maps")
                
        else:
            print("No metadata DataFrame found in drives object")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

def create_complete_trajectory_map(lats, lons, center_lat, center_lon, drive_id, drives_obj):
    """Create the complete trajectory map (current functionality)"""
    
    # Create the map
    m1 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add detailed trajectory as polyline
    trajectory_points = list(zip(lats, lons))
    
    folium.PolyLine(
        trajectory_points,
        weight=3,
        color='blue',
        opacity=0.4,
        popup=f'Complete GPS Trajectory<br>Drive ID: {drive_id}<br>{len(trajectory_points)} points'
    ).add_to(m1)
    
    # Add individual GPS points as small markers
    for i, (lat, lon) in enumerate(trajectory_points):
        if i == 0:
            # Start point
            folium.Marker(
                [lat, lon],
                popup=f'START<br>Drive ID: {drive_id}<br>Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}',
                icon=folium.Icon(color='green', icon='play'),
                tooltip=f'Start Point'
            ).add_to(m1)
        elif i == len(trajectory_points) - 1:
            # End point
            folium.Marker(
                [lat, lon],
                popup=f'END<br>Drive ID: {drive_id}<br>Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}',
                icon=folium.Icon(color='red', icon='stop'),
                tooltip=f'End Point'
            ).add_to(m1)
        else:
            # Show ALL intermediate points
            folium.CircleMarker(
                [lat, lon],
                radius=3,
                popup=f'GPS Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}',
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.6,
                tooltip=f'GPS Point {i+1}'
            ).add_to(m1)
    
    # Save complete trajectory map
    car_id = getattr(drives_obj, 'car_id', 'Unknown')
    map_file1 = f"complete_trip_car_{car_id}_drive_{drive_id}.html"
    m1.save(map_file1)
    print(f"Complete trajectory map saved as: {map_file1}")

def create_segmented_trajectory_map(detailed_trajectory, center_lat, center_lon, drive_id, drives_obj):
    """Create the segmented trajectory map with red boundary coordinates and no inter-segment lines"""
    
    # Create the map
    m2 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Create speed-based segments
    segments = create_speed_segments(detailed_trajectory)
    
    if segments is None or len(segments) == 0:
        print("No segments created - adding fallback complete trajectory")
        # Fallback to complete trajectory if no speed data
        lats = detailed_trajectory['latitude'].dropna().values
        lons = detailed_trajectory['longitude'].dropna().values
        trajectory_points = list(zip(lats, lons))
        
        folium.PolyLine(
            trajectory_points,
            weight=3,
            color='blue',
            opacity=0.8,
            popup=f'No Speed Data - Complete Trajectory<br>Drive ID: {drive_id}'
        ).add_to(m2)
    else:
        print(f"Drawing {len(segments)} completely separate segments...")
        
        # Add overall trip start marker (from first segment)
        if segments and len(segments[0]) > 0:
            first_point = segments[0][0]
            folium.Marker(
                [first_point[0], first_point[1]],
                popup=f'TRIP START<br>Drive ID: {drive_id}<br>Speed: {first_point[2]:.1f} km/h<br>Total segments: {len(segments)}',
                icon=folium.Icon(color='green', icon='play', prefix='fa'),
                tooltip=f'Trip Start'
            ).add_to(m2)
        
        # Add overall trip end marker (from last segment)
        if segments and len(segments[-1]) > 0:
            last_point = segments[-1][-1]
            folium.Marker(
                [last_point[0], last_point[1]],
                popup=f'TRIP END<br>Drive ID: {drive_id}<br>Speed: {last_point[2]:.1f} km/h<br>Total segments: {len(segments)}',
                icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                tooltip=f'Trip End'
            ).add_to(m2)
        
        # Draw each segment completely independently
        for seg_idx, segment in enumerate(segments):
            if len(segment) < 2:
                continue
            
            print(f"Drawing segment {seg_idx + 1} with {len(segment)} points")
            
            # Extract coordinates for this segment only
            segment_points = [(point[0], point[1]) for point in segment]
            
            # Draw this segment as a completely independent blue polyline
            folium.PolyLine(
                segment_points,
                weight=3,
                color='blue',
                opacity=0.8,
                popup=f'Segment {seg_idx + 1}<br>Points: {len(segment_points)}<br>Speed: {min(p[2] for p in segment):.1f} - {max(p[2] for p in segment):.1f} km/h'
            ).add_to(m2)
            
            # Paint ALL coordinates in this segment
            for point_idx, (lat, lon, speed, orig_idx) in enumerate(segment):
                # First and last points (segment boundaries) in RED - LARGER SIZE
                if point_idx == 0 or point_idx == len(segment) - 1:
                    folium.CircleMarker(
                        [lat, lon],
                        radius=8,  # Increased from 4 to 8
                        popup=f'Segment {seg_idx + 1} BOUNDARY<br>{"START" if point_idx == 0 else "END"}<br>Speed: {speed:.1f} km/h',
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.9,
                        tooltip=f'S{seg_idx + 1} {"Start" if point_idx == 0 else "End"}'
                    ).add_to(m2)
                
                # Middle points (movement) in BLUE
                else:
                    folium.CircleMarker(
                        [lat, lon],
                        radius=2,
                        popup=f'Segment {seg_idx + 1}<br>Point {point_idx + 1}/{len(segment)}<br>Speed: {speed:.1f} km/h',
                        color='blue',
                        fill=True,
                        fillColor='blue',
                        fillOpacity=0.6,
                        tooltip=f'S{seg_idx + 1}-P{point_idx + 1}'
                    ).add_to(m2)
    
    # Save segmented trajectory map
    car_id = getattr(drives_obj, 'car_id', 'Unknown')
    map_file2 = f"red_boundaries_car_{car_id}_drive_{drive_id}.html"
    m2.save(map_file2)
    print(f"Red boundaries map saved as: {map_file2}")
    
    # Print summary
    if segments:
        print(f"\nSegmentation summary:")
        print(f"- Total disconnected segments: {len(segments)}")
        for i, seg in enumerate(segments):
            start_speed = seg[0][2]
            end_speed = seg[-1][2]
            max_speed = max(p[2] for p in seg)
            print(f"  Segment {i+1}: {len(seg)} points, starts at {start_speed:.1f} km/h, ends at {end_speed:.1f} km/h, max {max_speed:.1f} km/h")
        print(f"\n🔴 Large red circles mark segment boundaries (start/end points)")
        print(f"🔵 Blue circles mark movement points within segments")
        print(f"🔵 Blue lines connect points ONLY within each segment")
        print(f"❌ NO lines connect different segments")
        print(f"🟢 Green marker shows overall trip start")
        print(f"🔴 Red marker shows overall trip end")

def calculate_total_distance(trajectory_points):
    """Calculate total distance along the trajectory"""
    total_dist = 0
    for i in range(1, len(trajectory_points)):
        dist = calculate_distance(
            trajectory_points[i-1][0], trajectory_points[i-1][1],
            trajectory_points[i][0], trajectory_points[i][1]
        )
        total_dist += dist
    return total_dist

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two GPS coordinates in kilometers"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

if __name__ == "__main__":
    visualize_random_trip() 