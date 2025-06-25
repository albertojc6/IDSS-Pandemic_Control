from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, SubmitField, IntegerField, FloatField
from wtforms.validators import DataRequired, Optional, NumberRange, InputRequired
from datetime import datetime

class CovidStatsForm(FlaskForm):
    state = SelectField('State', choices=[
        ('', 'All States'),
        ('AL', 'Alabama'),
        ('AK', 'Alaska'),
        ('AZ', 'Arizona'),
        ('AR', 'Arkansas'),
        ('CA', 'California'),
        ('CO', 'Colorado'),
        ('CT', 'Connecticut'),
        ('DE', 'Delaware'),
        ('FL', 'Florida'),
        ('GA', 'Georgia'),
        ('HI', 'Hawaii'),
        ('ID', 'Idaho'),
        ('IL', 'Illinois'),
        ('IN', 'Indiana'),
        ('IA', 'Iowa'),
        ('KS', 'Kansas'),
        ('KY', 'Kentucky'),
        ('LA', 'Louisiana'),
        ('ME', 'Maine'),
        ('MD', 'Maryland'),
        ('MA', 'Massachusetts'),
        ('MI', 'Michigan'),
        ('MN', 'Minnesota'),
        ('MS', 'Mississippi'),
        ('MO', 'Missouri'),
        ('MT', 'Montana'),
        ('NE', 'Nebraska'),
        ('NV', 'Nevada'),
        ('NH', 'New Hampshire'),
        ('NJ', 'New Jersey'),
        ('NM', 'New Mexico'),
        ('NY', 'New York'),
        ('NC', 'North Carolina'),
        ('ND', 'North Dakota'),
        ('OH', 'Ohio'),
        ('OK', 'Oklahoma'),
        ('OR', 'Oregon'),
        ('PA', 'Pennsylvania'),
        ('RI', 'Rhode Island'),
        ('SC', 'South Carolina'),
        ('SD', 'South Dakota'),
        ('TN', 'Tennessee'),
        ('TX', 'Texas'),
        ('UT', 'Utah'),
        ('VT', 'Vermont'),
        ('VA', 'Virginia'),
        ('WA', 'Washington'),
        ('WV', 'West Virginia'),
        ('WI', 'Wisconsin'),
        ('WY', 'Wyoming')
    ], validators=[Optional()])
    start_date = DateField('Start Date', format='%Y-%m-%d', validators=[Optional()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[Optional()])
    submit = SubmitField('Get Statistics')

class DailyStatsForm(FlaskForm):
    # Daily increase fields
    positiveIncrease = IntegerField('New Positive Cases', validators=[InputRequired(), NumberRange(min=0)])
    negativeIncrease = IntegerField('New Negative Cases', validators=[InputRequired(), NumberRange(min=0)])
    totalTestResultsIncrease = IntegerField('New Test Results', validators=[InputRequired(), NumberRange(min=0)])
    deathIncrease = IntegerField('New Deaths', validators=[InputRequired(), NumberRange(min=0)])
    hospitalizedIncrease = IntegerField('New Hospitalizations', validators=[InputRequired(), NumberRange(min=0)])
    
    # Vaccination data - only increases
    Dose1_Increase = IntegerField('New First Doses', validators=[InputRequired(), NumberRange(min=0)])
    Complete_Increase = IntegerField('New Completed Vaccinations', validators=[InputRequired(), NumberRange(min=0)])
    Dose1_65Plus_Increase = IntegerField('New First Doses (65+)', validators=[InputRequired(), NumberRange(min=0)])
    Complete_65Plus_Increase = IntegerField('New Completed Vaccinations (65+)', validators=[InputRequired(), NumberRange(min=0)])
    
    # Satisfaction rating
    satisfaction_rating = FloatField('System Satisfaction Rating (1-5)', 
                                   validators=[InputRequired(), NumberRange(min=1, max=5)],
                                   description='Rate your satisfaction with the system\'s recommendations and support')
    
    submit = SubmitField('Submit Daily Statistics')