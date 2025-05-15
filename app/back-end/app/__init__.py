from flask import Flask, redirect, url_for
from config import Config
from app.extensions import db, login_manager, bcrypt, migrate
import pandas as pd
from app.models import PandemicData, User, StaticStateData, Prediction
import os
from sqlalchemy import text
from pathlib import Path
from app.services.prophet_predictor import ProphetPredictor
from datetime import datetime, timedelta
from sqlalchemy import desc
from app.services.fuzzy_epidemiology import FuzzyEpidemiology

def reset_database(app):
    """
    Reset the database and reload all data.
    This will delete all existing data and recreate it from the CSV file.
    """
    with app.app_context():
        # Drop all tables
        db.drop_all()
        # Create all tables
        db.create_all()
        # Reload data
        load_initial_data(app)
        app.logger.info('Database has been reset and data reloaded')

def create_state_users(app):
    """
    Create a user account for each state if it doesn't exist.
    Username and password will be the state name.
    """
    with app.app_context():
        # Get unique states from the database
        states = db.session.query(PandemicData.state).distinct().all()
        states = [state[0] for state in states]  # Convert from list of tuples to list of strings

        for state in states:
            # Check if user already exists
            user = User.query.filter_by(username=state).first()
            if user is None:
                # Create new user for the state
                user = User(
                    username=state,
                    email=f"{state.lower().replace(' ', '_')}@example.com",
                    state_name=state
                )
                user.set_password(state)  # Set password same as state name
                db.session.add(user)
                app.logger.info(f'Created user account for state: {state}')
        
        db.session.commit()

def load_static_state_data(app):
    """
    Load static state-level data from the CSV file if the database is empty.
    """
    with app.app_context():
        # Check if we already have data
        if StaticStateData.query.first() is None:
            try:
                # Get the path to the CSV file
                base_path = Path(__file__).parent.parent
                csv_path = base_path / "data" / "preprocessed" / "dataMatrix" / "static_stateMatrix.csv"
                
                if csv_path.exists():
                    # Read the CSV file
                    df = pd.read_csv(csv_path)
                    
                    # Process each row and create StaticStateData entries
                    records_added = 0
                    for _, row in df.iterrows():
                        data = StaticStateData(
                            state=row['state'],
                            no_coverage=row['no_coverage'],
                            private_coverage=row['private_coverage'],
                            public_coverage=row['public_coverage'],
                            labor_cov_diff=row['labor_cov_diff'],
                            bedsState_local_government=row['bedsState_local_government'],
                            bedsNon_profit=row['bedsNon_profit'],
                            bedsFor_profit=row['bedsFor_profit'],
                            bedsTotal=row['bedsTotal'],
                            population_state=row['population_state'],
                            pop_density_state=row['pop_density_state'],
                            pop_0_9=row['pop_0-9'],
                            pop_10_19=row['pop_10-19'],
                            pop_20_29=row['pop_20-29'],
                            pop_30_39=row['pop_30-39'],
                            pop_40_49=row['pop_40-49'],
                            pop_50_59=row['pop_50-59'],
                            pop_60_69=row['pop_60-69'],
                            pop_70_79=row['pop_70-79'],
                            pop_80_plus=row['pop_80+'],
                            Low_SVI_CTGY=row['Low_SVI_CTGY'],
                            Moderate_Low_SVI_CTGY=row['Moderate_Low_SVI_CTGY'],
                            Moderate_High_SVI_CTGY=row['Moderate_High_SVI_CTGY'],
                            High_SVI_CTGY=row['High_SVI_CTGY'],
                            Metro=row['Metro'],
                            Non_metro=row['Non-metro']
                        )
                        db.session.add(data)
                        records_added += 1
                    
                    db.session.commit()
                    app.logger.info(f'Successfully imported {records_added} records from static_stateMatrix.csv')
                else:
                    app.logger.warning('static_stateMatrix.csv not found in data directory')
            except Exception as e:
                app.logger.error(f'Error importing static state data: {str(e)}')
                db.session.rollback()

def load_initial_data(app):
    """
    Load initial COVID-19 data and static state data from the CSV files if the database is empty.
    """
    with app.app_context():
        # Load pandemic data
        if PandemicData.query.first() is None:
            try:
                # Get the path to the CSV file
                base_path = Path(__file__).parent.parent
                csv_path = base_path / "data" / "preprocessed" / "dataMatrix" / "daily_covidMatrix.csv"
                
                if csv_path.exists():
                    # Read the CSV file
                    df = pd.read_csv(csv_path)
                    
                    # Convert date column to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Process each row and create PandemicData entries
                    records_added = 0
                    for _, row in df.iterrows():
                        data = PandemicData(
                            date=row['date'].date(),
                            state=row['state'],
                            positive=row.get('positive', 0),
                            totalTestResults=row.get('totalTestResults', 0),
                            death=row.get('death', 0),
                            positiveIncrease=row.get('positiveIncrease', 0),
                            negativeIncrease=row.get('negativeIncrease', 0),
                            total=row.get('total', 0),
                            totalTestResultsIncrease=row.get('totalTestResultsIncrease', 0),
                            posNeg=row.get('posNeg', 0),
                            deathIncrease=row.get('deathIncrease', 0),
                            hospitalizedIncrease=row.get('hospitalizedIncrease', 0),
                            Dose1_Total=row.get('Dose1_Total', 0),
                            Dose1_Total_pct=row.get('Dose1_Total_pct', 0.0),
                            Dose1_65Plus=row.get('Dose1_65Plus', 0),
                            Dose1_65Plus_pct=row.get('Dose1_65Plus_pct', 0.0),
                            Complete_Total=row.get('Complete_Total', 0),
                            Complete_Total_pct=row.get('Complete_Total_pct', 0.0),
                            Complete_65Plus=row.get('Complete_65Plus', 0),
                            Complete_65Plus_pct=row.get('Complete_65Plus_pct', 0.0)
                        )
                        db.session.add(data)
                        records_added += 1
                    
                    db.session.commit()
                    app.logger.info(f'Successfully imported {records_added} records from daily_covidMatrix.csv')
                    
                    # Create users for each state after data is imported
                    create_state_users(app)
                else:
                    app.logger.warning('daily_covidMatrix.csv not found in data directory')
            except Exception as e:
                app.logger.error(f'Error importing pandemic data: {str(e)}')
                db.session.rollback()
        
        # Load static state data
        load_static_state_data(app)

def generate_initial_predictions(app):
    """
    Generate initial predictions and recommendations for all states using their latest available data.
    """
    with app.app_context():
        predictor = app.prophet_predictor
        fuzzy_system = FuzzyEpidemiology()
        
        # Get list of states
        states = [state[0] for state in PandemicData.query.with_entities(PandemicData.state).distinct().all()]
        
        # Make predictions and recommendations for each state
        for state in states:
            try:
                # Get the latest date for this specific state
                latest_state_date = PandemicData.query.filter_by(
                    state=state
                ).order_by(desc(PandemicData.date)).first()
                
                if not latest_state_date:
                    app.logger.warning(f"No data found for state {state}")
                    continue
                    
                latest_date = latest_state_date.date
                latest_datetime = datetime.combine(latest_date, datetime.min.time())
                
                # Check if prediction already exists for this state and date
                existing_prediction = Prediction.query.filter_by(
                    state=state,
                    date=latest_date
                ).first()
                
                if not existing_prediction:
                    # Generate prediction
                    prediction = predictor.predict_for_state(state, latest_datetime)
                    db.session.add(prediction)
                    db.session.commit()
                    app.logger.info(f'Generated prediction for {state} using data from {latest_date}')
                    
                    # Generate recommendation
                    try:
                        recommendation = fuzzy_system.get_knowledge(state, latest_date)
                        app.logger.info(f'Generated recommendation for {state} with risk level: {recommendation.risk_level}')
                        print(f'Generated recommendation for {state}:')
                        print(f'  - Risk Level: {recommendation.risk_level}')
                        print(f'  - Confinement: {recommendation.confinement_level}')
                        print(f'  - Beds: {recommendation.beds_recommendation}')
                        print(f'  - Vaccination: {recommendation.vaccination_percentage}%')
                    except Exception as e:
                        app.logger.error(f'Error generating recommendation for {state}: {str(e)}')
                        db.session.rollback()
            except Exception as e:
                app.logger.error(f'Error making prediction for {state}: {str(e)}')
                db.session.rollback()

def create_app(config_class=Config):
    """
    Application factory function that creates and configures the Flask application.
    
    Args:
        config_class: Configuration class to use for the application
        
    Returns:
        Flask application instance
    """
    # Create Flask app with custom template and static folders
    app = Flask(__name__,
                template_folder=config_class.TEMPLATES_FOLDER,
                static_folder=config_class.STATIC_FOLDER)
    app.config.from_object(config_class)
    
    # Initialize Flask extensions
    db.init_app(app)  # Database
    migrate.init_app(app, db)
    login_manager.init_app(app)  # User authentication
    bcrypt.init_app(app)  # Password hashing
    
    # Register blueprints for different sections of the application
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')  # Authentication routes
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)  # Main application routes
    
    from app.stats import bp as stats_bp
    app.register_blueprint(stats_bp, url_prefix='/stats')  # Statistics routes
    
    # Import models after extensions are initialized to avoid circular imports
    from app import models
    
    # Create database tables and load initial data
    with app.app_context():
        # Create all tables first
        db.create_all()
        
        # Load initial data from CSV files
        load_initial_data(app)
        
        # Initialize ProphetPredictor after data is loaded
        predictor = ProphetPredictor()
        predictor.load_models()
        app.prophet_predictor = predictor
        
        # Generate initial predictions
        generate_initial_predictions(app)
    
    # Root route that redirects to dashboard
    @app.route('/')
    def index():
        return redirect(url_for('main.dashboard'))
    
    # Add a route to reset the database (for development purposes)
    @app.route('/reset-db')
    def reset_db():
        if app.debug:  # Only allow in debug mode
            reset_database(app)
            return 'Database has been reset'
        return 'Reset not allowed in production mode', 403
    
    return app