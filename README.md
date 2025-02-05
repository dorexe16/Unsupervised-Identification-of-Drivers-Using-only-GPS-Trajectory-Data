# Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data

# Driver Identification Using GPS Data

## Overview
This project focuses on driver identification using only GPS trajectory data. Unlike other approaches that incorporate multi-modal data from in-vehicle sensors or cameras, this framework aims to provide an effective solution relying solely on GPS signals. The primary goal is to cluster and identify drivers based on their driving patterns without prior knowledge of the vehicle.

## Key Features
- **Pattern-Based Driver Identification**: Utilizing repetitive trip patterns to segment and analyze trips.
- **MiniRocket Transformation**: A fast and scalable feature extraction method applied to GPS-based time series.
- **Clustering & Outlier Detection**: DBSCAN and Spectral Clustering techniques to identify meaningful driver groups.
- **Scaling & Normalization**: Removing environmental bias by considering road speed limits and traffic conditions.
- **Experimentation & Analysis**: Evaluating different clustering strategies and feature representations.

## Important Clarification
This project **is not focused on active learning**. While active learning is explored in some experiments, the core purpose of this research is to **identify drivers using GPS data through feature engineering, transformation, and clustering techniques**. The active learning component is secondary and used to refine predictions in some cases but is not the primary research goal.

## Project Structure
```
driver-identification/
│── data/                     # (Optional) Sample data or data processing scripts
│── notebooks/                # Jupyter notebooks for exploration and results
│── src/                      # Source code
│   ├── preprocessing/        # Data preparation and feature extraction
│   ├── modeling/             # Training, clustering, and feature transformations
│   ├── utils/                # Helper functions (e.g., geolocation, transformations)
│── experiments/              # Experimentation scripts
│── docs/                     # Documentation, images, and methodology
│── tests/                    # Unit tests for robustness
│── .gitignore                # Ignore unnecessary files
│── README.md                 # Overview of the project
│── requirements.txt          # Dependencies
│── setup.py                  # Installation script (if needed)
```

## Installation
To set up the project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Preprocess Data**: Convert raw GPS data into structured trip data.
2. **Feature Extraction**: Apply transformations using MiniRocket and PCA.
3. **Clustering & Identification**: Use unsupervised clustering techniques to identify unique drivers.
4. **Evaluation**: Analyze clustering performance and experiment with different feature sets.

## Dataset
The dataset consists of naturalistic GPS trajectories from multiple vehicles. Each trip is represented as a time series with features such as:
- **Time Stamps**
- **Latitude & Longitude**
- **Speed & Acceleration**
- **Direction Changes**

## Results & Insights
- **Driver clustering achieved high accuracy** using MiniRocket features combined with Spectral Clustering.
- **Environmental scaling significantly improved clustering reliability** by reducing road-specific biases.
- **Pattern-based segmentation proved effective**, as trips with consistent destinations showed strong homogeneity in driver identity.

## Future Work
- Improve clustering robustness for highly variable routes.
- Investigate alternative feature representations for driver identification.
- Explore scalability on larger datasets with more diverse driving behaviors.

## Contributors
- **Dor Bar**  
- **Irad Ben Gal**  

## License
This project is open-source under the MIT License. Feel free to contribute!

