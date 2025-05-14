# IDSS-Pandemic Control

A web application for monitoring and predicting COVID-19 trends across US states using machine learning models.

## Features

- Real-time COVID-19 data visualization for US states
- Prophet-based prediction models for:
  - Positive cases
  - Hospitalizations
  - Deaths
- State-based user authentication
- Interactive dashboard with statistical analysis
- Hierarchical clustering of states based on COVID-19 patterns

## Project Structure

```
app/
├── back-end/           # Flask application
│   ├── data/          # Data storage
│   │   ├── preprocessed/
│   │   │   └── dataMatrix/
│   │   │       ├── daily_covidMatrix.csv
│   │   │       └── static_stateMatrix.csv
│   │   ├── models/    # Trained Prophet models
│   │   └── clustering/
│   │       └── state_clusters.csv
│   └── app/           # Application code
└── front-end/         # React frontend
    ├── templates/     # HTML templates
    └── static/        # Static assets
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

3. Run the application:
```bash
python back-end/run.py
```

## Data Sources

- COVID-19 data from official state sources
- State-level demographic and healthcare data
- Vaccination statistics

## Technologies

- Backend: Flask, SQLAlchemy, Prophet
- Frontend: React, Chart.js
- Database: SQLite
- Machine Learning: Prophet, scikit-learn