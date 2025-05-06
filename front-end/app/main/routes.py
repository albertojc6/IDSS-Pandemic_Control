from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.models import PandemicData
from app.utils.data_generator import get_state_abbreviation
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from collections import defaultdict
from app.main import bp  # Changed to import bp from main blueprint

@bp.route('/')  # Changed from main to bp
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@bp.route('/dashboard')  # Changed from main to bp
@login_required
def dashboard():
    # Get the latest data for each state
    latest_data_query = PandemicData.query.with_entities(
        PandemicData.state, 
        func.max(PandemicData.date).label('max_date')
    ).group_by(PandemicData.state).subquery()
    
    latest_data = PandemicData.query.join(
        latest_data_query,
        (PandemicData.state == latest_data_query.c.state) & 
        (PandemicData.date == latest_data_query.c.max_date)
    ).all()
    
    # Get national data (sum of all states for the latest date)
    latest_date = PandemicData.query.with_entities(func.max(PandemicData.date)).scalar()
    
    if latest_date:
        national_data = PandemicData.query.filter_by(date=latest_date).all()
        
        # Calculate national totals
        total_cases = sum(data.positive for data in national_data)
        total_deaths = sum(data.death for data in national_data)
        total_tests = sum(data.totalTestResults for data in national_data)
        
        # Estimate active cases and recovered (simplified calculation)
        active_cases = int(total_cases * 0.15)  # Assume 15% of total cases are active
        recovered = total_cases - active_cases - total_deaths
        
        # Get daily cases for the past 30 days
        thirty_days_ago = latest_date - timedelta(days=30)
        daily_data = PandemicData.query.filter(
            PandemicData.date >= thirty_days_ago
        ).order_by(PandemicData.date).all()
        
        # Aggregate daily cases by date for the entire country
        daily_cases_by_date = defaultdict(int)
        for data in daily_data:
            daily_cases_by_date[data.date.strftime('%Y-%m-%d')] += data.positiveIncrease
        
        national_daily_cases = [
            {'date': date, 'cases': cases} 
            for date, cases in sorted(daily_cases_by_date.items())
        ]
        
        # Prepare data for the map
        covid_data = {}
        for data in latest_data:
            # Calculate risk level based on positive rate
            positive_rate = data.positive / data.totalTestResults if data.totalTestResults > 0 else 0
            
            if positive_rate < 0.05:
                risk_level = "Low"
            elif positive_rate < 0.1:
                risk_level = "Medium"
            elif positive_rate < 0.15:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            # Estimate active cases (simplified)
            active_cases_state = int(data.positive * 0.15)
            
            covid_data[data.state] = {
                'total_cases': data.positive,
                'active_cases': active_cases_state,
                'deaths': data.death,
                'risk_level': risk_level,
                'abbr': get_state_abbreviation(data.state)
            }
        
        # Create a class-like object for national data
        class NationalData:
            def __init__(self, total_cases, active_cases, deaths, recovered):
                self.total_cases = total_cases
                self.active_cases = active_cases
                self.deaths = deaths
                self.recovered = recovered
        
        national_data_obj = NationalData(
            total_cases=total_cases,
            active_cases=active_cases,
            deaths=total_deaths,
            recovered=recovered
        )
        
        return render_template(
            'main/dashboard.html', 
            covid_data=covid_data,
            national_data=national_data_obj,
            national_daily_cases=national_daily_cases
        )
    
    # If no data is available
    return render_template('main/dashboard.html', covid_data={}, national_daily_cases=[])