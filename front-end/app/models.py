from datetime import datetime
from flask_login import UserMixin
from app.extensions import db, login_manager, bcrypt

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    state_name = db.Column(db.String(50), default="New York")  # Added state_name field
    last_login = db.Column(db.DateTime)  # Added last_login field
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class PandemicData(db.Model):
    __tablename__ = 'pandemic_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    state = db.Column(db.String(50), nullable=False)
    positive = db.Column(db.Integer)
    totalTestResults = db.Column(db.Integer)
    death = db.Column(db.Integer)
    positiveIncrease = db.Column(db.Integer)
    negativeIncrease = db.Column(db.Integer)
    total = db.Column(db.Integer)
    totalTestResultsIncrease = db.Column(db.Integer)
    posNeg = db.Column(db.Integer)
    deathIncrease = db.Column(db.Integer)
    hospitalizedIncrease = db.Column(db.Integer)
    Dose1_Total = db.Column(db.Integer)
    Dose1_Total_pct = db.Column(db.Float)
    Dose1_65Plus = db.Column(db.Integer)
    Dose1_65Plus_pct = db.Column(db.Float)
    Complete_Total = db.Column(db.Integer)
    Complete_Total_pct = db.Column(db.Float)
    Complete_65Plus = db.Column(db.Integer)
    Complete_65Plus_pct = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PandemicData {self.state} {self.date}>'
    
    def to_dict(self):
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))