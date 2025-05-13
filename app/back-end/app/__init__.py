from flask import Flask, redirect, url_for
from config import Config
from app.extensions import db, login_manager, bcrypt
import pandas as pd
from app.models import PandemicData, User
import os

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

def load_initial_data(app):
    """
    Load initial COVID-19 data from the CSV file if the database is empty.
    """
    with app.app_context():
        # Check if we already have data
        if PandemicData.query.first() is None:
            try:
                # Get the path to the CSV file
                csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'daily_covidMatrix.csv')
                
                if os.path.exists(csv_path):
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
                app.logger.error(f'Error importing data: {str(e)}')
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
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        # Load initial data from CSV
        load_initial_data(app)
    
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