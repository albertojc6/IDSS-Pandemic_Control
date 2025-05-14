from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, SubmitField, IntegerField, FloatField
from wtforms.validators import DataRequired, Optional, NumberRange
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
    positiveIncrease = IntegerField('New Positive Cases', validators=[DataRequired(), NumberRange(min=0)])
    negativeIncrease = IntegerField('New Negative Cases', validators=[DataRequired(), NumberRange(min=0)])
    totalTestResultsIncrease = IntegerField('New Test Results', validators=[DataRequired(), NumberRange(min=0)])
    deathIncrease = IntegerField('New Deaths', validators=[DataRequired(), NumberRange(min=0)])
    hospitalizedIncrease = IntegerField('New Hospitalizations', validators=[DataRequired(), NumberRange(min=0)])
    
    # Vaccination data
    Dose1_Total = IntegerField('Total First Doses', validators=[DataRequired(), NumberRange(min=0)])
    Dose1_Total_pct = FloatField('First Doses %', validators=[DataRequired(), NumberRange(min=0, max=100)])
    Dose1_65Plus = IntegerField('First Doses 65+', validators=[DataRequired(), NumberRange(min=0)])
    Dose1_65Plus_pct = FloatField('First Doses 65+ %', validators=[DataRequired(), NumberRange(min=0, max=100)])
    Complete_Total = IntegerField('Total Completed Vaccinations', validators=[DataRequired(), NumberRange(min=0)])
    Complete_Total_pct = FloatField('Completed Vaccinations %', validators=[DataRequired(), NumberRange(min=0, max=100)])
    Complete_65Plus = IntegerField('Completed Vaccinations 65+', validators=[DataRequired(), NumberRange(min=0)])
    Complete_65Plus_pct = FloatField('Completed Vaccinations 65+ %', validators=[DataRequired(), NumberRange(min=0, max=100)])
    
    submit = SubmitField('Submit Daily Statistics')