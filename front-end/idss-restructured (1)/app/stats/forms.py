from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional

class CovidStatsForm(FlaskForm):
    new_cases = IntegerField('New Cases', validators=[DataRequired(), NumberRange(min=0)])
    total_cases = IntegerField('Total Cases', validators=[DataRequired(), NumberRange(min=0)])
    new_deaths = IntegerField('New Deaths', validators=[DataRequired(), NumberRange(min=0)])
    total_deaths = IntegerField('Total Deaths', validators=[DataRequired(), NumberRange(min=0)])
    new_vaccinations = IntegerField('New Vaccinations', validators=[Optional(), NumberRange(min=0)])
    total_vaccinations = IntegerField('Total Vaccinations', validators=[Optional(), NumberRange(min=0)])
    current_hospitalizations = IntegerField('Current Hospitalizations', validators=[Optional(), NumberRange(min=0)])
    icu_patients = IntegerField('Current ICU Patients', validators=[Optional(), NumberRange(min=0)])
    submit = SubmitField('Submit Statistics')
