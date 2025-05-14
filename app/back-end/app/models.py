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
            'created_at': self.created_at.isoformat()
        }

@login_manager.user_loader
def load_user(user_id):
    """
    User loader callback for Flask-Login.
    Required to load the user from the user ID stored in the session.
    """
    return User.query.get(int(user_id))