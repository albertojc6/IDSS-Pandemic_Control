from datetime import datetime
from flask_login import UserMixin
from app.extensions import db, login_manager, bcrypt

class User(UserMixin, db.Model):
    """
    User model for authentication and user management.
    Inherits from UserMixin to provide Flask-Login functionality.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    state_name = db.Column(db.String(50), default="New York")  # User's associated state
    last_login = db.Column(db.DateTime)  # Track user's last login time
    
    def set_password(self, password):
        """Hash and set the user's password."""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        
    def check_password(self, password):
        """Verify the user's password."""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class PandemicData(db.Model):
    """
    Model for storing pandemic statistics and data.
    Contains comprehensive information about COVID-19 cases, deaths, tests, and vaccinations.
    """
    __tablename__ = 'pandemic_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    state = db.Column(db.String(50), nullable=False)
    positive = db.Column(db.Integer)  # Total positive cases
    totalTestResults = db.Column(db.Integer)  # Total test results
    death = db.Column(db.Integer)  # Total deaths
    positiveIncrease = db.Column(db.Integer)  # New positive cases
    negativeIncrease = db.Column(db.Integer)  # New negative cases
    total = db.Column(db.Integer)  # Total cases
    totalTestResultsIncrease = db.Column(db.Integer)  # New test results
    posNeg = db.Column(db.Integer)  # Positive/Negative ratio
    deathIncrease = db.Column(db.Integer)  # New deaths
    hospitalizedIncrease = db.Column(db.Integer)  # New hospitalizations
    Dose1_Total = db.Column(db.Integer)  # Total first doses
    Dose1_Total_pct = db.Column(db.Float)  # Percentage of first doses
    Dose1_65Plus = db.Column(db.Integer)  # First doses for 65+
    Dose1_65Plus_pct = db.Column(db.Float)  # Percentage of first doses for 65+
    Complete_Total = db.Column(db.Integer)  # Total completed vaccinations
    Complete_Total_pct = db.Column(db.Float)  # Percentage of completed vaccinations
    Complete_65Plus = db.Column(db.Integer)  # Completed vaccinations for 65+
    Complete_65Plus_pct = db.Column(db.Float)  # Percentage of completed vaccinations for 65+
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Record creation timestamp
    
    def __repr__(self):
        return f'<PandemicData {self.state} {self.date}>'
    
    def to_dict(self):
        """
        Convert the model instance to a dictionary.
        Used for JSON serialization in API responses.
        """
        return {
            'id': self.id,
            'date': self.date.strftime('%Y-%m-%d') if self.date else None,
            'state': self.state,
            'positive': self.positive,
            'totalTestResults': self.totalTestResults,
            'death': self.death,
            'positiveIncrease': self.positiveIncrease,
            'negativeIncrease': self.negativeIncrease,
            'total': self.total,
            'totalTestResultsIncrease': self.totalTestResultsIncrease,
            'posNeg': self.posNeg,
            'deathIncrease': self.deathIncrease,
            'hospitalizedIncrease': self.hospitalizedIncrease,
            'Dose1_Total': self.Dose1_Total,
            'Dose1_Total_pct': self.Dose1_Total_pct,
            'Dose1_65Plus': self.Dose1_65Plus,
            'Dose1_65Plus_pct': self.Dose1_65Plus_pct,
            'Complete_Total': self.Complete_Total,
            'Complete_Total_pct': self.Complete_Total_pct,
            'Complete_65Plus': self.Complete_65Plus,
            'Complete_65Plus_pct': self.Complete_65Plus_pct
        }

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(2), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Target variables (7-day sums)
    positive_increase_sum = db.Column(db.Integer, nullable=False)
    hospitalized_increase_sum = db.Column(db.Integer, nullable=False)
    death_increase_sum = db.Column(db.Integer, nullable=False)
    
    # Daily predictions (stored as JSON strings)
    positive_daily = db.Column(db.JSON, nullable=False)  # List of 7 daily predictions
    hospitalized_daily = db.Column(db.JSON, nullable=False)  # List of 7 daily predictions
    death_daily = db.Column(db.JSON, nullable=False)  # List of 7 daily predictions
    
    # Metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.state} {self.date}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'state': self.state,
            'date': self.date.isoformat(),
            'positive_increase_sum': self.positive_increase_sum,
            'hospitalized_increase_sum': self.hospitalized_increase_sum,
            'death_increase_sum': self.death_increase_sum,
            'positive_daily': self.positive_daily,
            'hospitalized_daily': self.hospitalized_daily,
            'death_daily': self.death_daily,
            'created_at': self.created_at.isoformat()
        }

class StaticStateData(db.Model):
    """
    Model for storing static state-level data and characteristics.
    Contains information about healthcare coverage, hospital beds, population demographics, and social vulnerability.
    """
    __tablename__ = 'static_state_data'
    
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(50), nullable=False, unique=True)
    
    # Healthcare coverage
    no_coverage = db.Column(db.Float)  # Percentage without coverage
    private_coverage = db.Column(db.Float)  # Percentage with private coverage
    public_coverage = db.Column(db.Float)  # Percentage with public coverage
    labor_cov_diff = db.Column(db.Float)  # Labor coverage difference
    
    # Hospital beds per 1000 population
    bedsState_local_government = db.Column(db.Float)
    bedsNon_profit = db.Column(db.Float)
    bedsFor_profit = db.Column(db.Float)
    bedsTotal = db.Column(db.Float)
    
    # Population data
    population_state = db.Column(db.Integer)
    pop_density_state = db.Column(db.Float)
    
    # Age demographics (percentages)
    pop_0_9 = db.Column(db.Float)
    pop_10_19 = db.Column(db.Float)
    pop_20_29 = db.Column(db.Float)
    pop_30_39 = db.Column(db.Float)
    pop_40_49 = db.Column(db.Float)
    pop_50_59 = db.Column(db.Float)
    pop_60_69 = db.Column(db.Float)
    pop_70_79 = db.Column(db.Float)
    pop_80_plus = db.Column(db.Float)
    
    # Social Vulnerability Index categories (percentages)
    Low_SVI_CTGY = db.Column(db.Float)
    Moderate_Low_SVI_CTGY = db.Column(db.Float)
    Moderate_High_SVI_CTGY = db.Column(db.Float)
    High_SVI_CTGY = db.Column(db.Float)
    
    # Urban/Rural classification (percentages)
    Metro = db.Column(db.Float)
    Non_metro = db.Column(db.Float)
    
    def __repr__(self):
        return f'<StaticStateData {self.state}>'
    
    def to_dict(self):
        """
        Convert the model instance to a dictionary.
        Used for JSON serialization in API responses.
        """
        return {
            'id': self.id,
            'state': self.state,
            'no_coverage': self.no_coverage,
            'private_coverage': self.private_coverage,
            'public_coverage': self.public_coverage,
            'labor_cov_diff': self.labor_cov_diff,
            'bedsState_local_government': self.bedsState_local_government,
            'bedsNon_profit': self.bedsNon_profit,
            'bedsFor_profit': self.bedsFor_profit,
            'bedsTotal': self.bedsTotal,
            'population_state': self.population_state,
            'pop_density_state': self.pop_density_state,
            'pop_0_9': self.pop_0_9,
            'pop_10_19': self.pop_10_19,
            'pop_20_29': self.pop_20_29,
            'pop_30_39': self.pop_30_39,
            'pop_40_49': self.pop_40_49,
            'pop_50_59': self.pop_50_59,
            'pop_60_69': self.pop_60_69,
            'pop_70_79': self.pop_70_79,
            'pop_80_plus': self.pop_80_plus,
            'Low_SVI_CTGY': self.Low_SVI_CTGY,
            'Moderate_Low_SVI_CTGY': self.Moderate_Low_SVI_CTGY,
            'Moderate_High_SVI_CTGY': self.Moderate_High_SVI_CTGY,
            'High_SVI_CTGY': self.High_SVI_CTGY,
            'Metro': self.Metro,
            'Non_metro': self.Non_metro
        }

class Recommendation(db.Model):
    """Model for storing state recommendations and metrics"""
    __tablename__ = 'recommendations'

    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(2), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Predictions
    infected = db.Column(db.Integer, nullable=False)
    hospitalized = db.Column(db.Integer, nullable=False)
    deaths = db.Column(db.Integer, nullable=False)
    
    # Metrics
    ia = db.Column(db.Float, nullable=False)  # Cumulative incidence
    theta = db.Column(db.Float, nullable=False)  # Hospital occupancy
    pi = db.Column(db.Float, nullable=False)  # Mortality rate
    lethality = db.Column(db.Float, nullable=False)
    pop_over_65 = db.Column(db.Float, nullable=False)
    density = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.Float, nullable=False)
    
    # Recommendations
    beds_recommendation = db.Column(db.String(255), nullable=False)
    vaccination_percentage = db.Column(db.Float, nullable=False)
    confinement_level = db.Column(db.String(50), nullable=False)

    def to_dict(self) -> dict:
        """Convert recommendation to dictionary format"""
        return {
            'state': self.state,
            'date': self.date.isoformat(),
            'predictions': {
                'infected': self.infected,
                'hospitalized': self.hospitalized,
                'deaths': self.deaths
            },
            'metrics': {
                'IA': self.ia,
                'theta': self.theta,
                'pi': self.pi,
                'lethality': self.lethality,
                'pop_>65': self.pop_over_65,
                'density': self.density,
                'risk_level': self.risk_level
            },
            'recommendations': {
                'beds': [None, self.beds_recommendation],
                'vaccination': [None, self.vaccination_percentage],
                'confinement': [None, self.confinement_level]
            }
        }

    def __repr__(self):
        return f'<Recommendation {self.state} {self.date}>'

@login_manager.user_loader
def load_user(user_id):
    """
    User loader callback for Flask-Login.
    Required to load the user from the user ID stored in the session.
    """
    return User.query.get(int(user_id))