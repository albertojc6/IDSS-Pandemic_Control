from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from app.models import PandemicData, Prediction
from app.utils.data_generator import get_state_abbreviation
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from collections import defaultdict
from app.main import bp  # Changed to import bp from main blueprint
from app.extensions import db
from app.services.prophet_predictor import ProphetPredictor

@bp.route('/')  # Changed from main to bp
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

def update_predictions():
    """Update predictions for all states"""
    predictor = current_app.prophet_predictor
    
    # Get list of states
    states = [state[0] for state in PandemicData.query.with_entities(PandemicData.state).distinct().all()]
    
    # Get latest date in the database and convert to datetime
    latest_date = PandemicData.query.with_entities(PandemicData.date).order_by(desc(PandemicData.date)).first()[0]
    latest_datetime = datetime.combine(latest_date, datetime.min.time())
    
    # Get the latest data for the logged-in state
    latest_state_data = PandemicData.query.filter_by(
        state=current_user.state_name,
        date=latest_date
    ).first()
    
    if latest_state_data:
        print(f"\nMaking predictions for {current_user.state_name} using data from {latest_date}")
        print("=" * 80)
        print(f"Date: {latest_state_data.date}")
        print(f"Positive Cases: {latest_state_data.positive}")
        print(f"Positive Increase: {latest_state_data.positiveIncrease}")
        print(f"Hospitalized Increase: {latest_state_data.hospitalizedIncrease}")
        print(f"Death Increase: {latest_state_data.deathIncrease}")
        print(f"Total Tests: {latest_state_data.totalTestResults}")
        print("=" * 80)
    
    # Make predictions for each state
    for state in states:
        try:
            # Check if prediction already exists for this state and date
            existing_prediction = Prediction.query.filter_by(
                state=state,
                date=latest_date
            ).first()
            
            if not existing_prediction:
                prediction = predictor.predict_for_state(state, latest_datetime)
                db.session.add(prediction)
                db.session.commit()
        except Exception as e:
            current_app.logger.error(f"Error making prediction for {state}: {str(e)}")
            if state == current_user.state_name:
                print(f"Error making prediction for {state}: {str(e)}")

@bp.route('/dashboard')
@login_required
def dashboard():
    # Update predictions before showing dashboard
    update_predictions()
    
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
    
    # Get national data by summing the latest data from each state
    national_data = latest_data  # Use the latest data from each state
    
    if national_data:
        # Calculate national totals
        total_cases = sum(data.positive for data in national_data)
        total_deaths = sum(data.death for data in national_data)
        
        # Get data for the last 14 days to calculate active and recovered cases
        latest_date = max(data.date for data in national_data)
        fourteen_days_ago = latest_date - timedelta(days=14)
        
        # Get daily data for the last 14 days
        daily_data = PandemicData.query.filter(
            PandemicData.date >= fourteen_days_ago
        ).order_by(PandemicData.date).all()
        
        # Calculate active and recovered cases
        active_cases = 0
        recovered = 0
        
        # Group data by state
        state_data = {}
        for data in daily_data:
            if data.state not in state_data:
                state_data[data.state] = []
            state_data[data.state].append(data)
        
        # Calculate for each state
        for state, state_records in state_data.items():
            # Sort records by date
            state_records.sort(key=lambda x: x.date)
            
            # Calculate active cases (cases from last 14 days)
            state_active = sum(record.positiveIncrease for record in state_records)
            
            # Calculate recovered (total cases - active cases - deaths)
            state_total = state_records[-1].positive
            state_deaths = state_records[-1].death
            state_recovered = state_total - state_active - state_deaths
            
            active_cases += state_active
            recovered += state_recovered
        
        # Get state-specific data for the logged-in user
        state_data = None
        for data in national_data:
            if data.state.lower() == current_user.state_name.lower():
                state_data = data
                break
        
        # Get latest prediction for the user's state
        latest_prediction = Prediction.query.filter_by(
            state=current_user.state_name
        ).order_by(desc(Prediction.date)).first()
        
        if state_data:
            # Get state's daily data for the last 14 days
            state_daily_data = PandemicData.query.filter(
                PandemicData.date >= fourteen_days_ago,
                PandemicData.state == state_data.state
            ).order_by(PandemicData.date).all()
            
            # Calculate state's active cases
            state_active_cases = sum(record.positiveIncrease for record in state_daily_data)
            
            # Calculate state's recovered cases
            state_total_cases = state_data.positive
            state_deaths = state_data.death
            state_recovered = state_total_cases - state_active_cases - state_deaths
        else:
            state_total_cases = 0
            state_deaths = 0
            state_active_cases = 0
            state_recovered = 0
        
        # Get daily cases for the past 30 days
        latest_date = PandemicData.query.with_entities(func.max(PandemicData.date)).scalar()
        thirty_days_ago = latest_date - timedelta(days=30)
        daily_data = PandemicData.query.filter(
            PandemicData.date >= thirty_days_ago
        ).order_by(PandemicData.date).all()
        
        # Aggregate daily data by date for the entire country and the logged-in state
        daily_cases_by_date = defaultdict(lambda: {'usa_cases': 0, 'usa_deaths': 0, 'state_cases': 0, 'state_deaths': 0})
        
        for data in daily_data:
            date_str = data.date.strftime('%Y-%m-%d')
            # Add to USA totals
            daily_cases_by_date[date_str]['usa_cases'] += data.positiveIncrease
            daily_cases_by_date[date_str]['usa_deaths'] += data.deathIncrease
            
            # Add to state totals if it's the logged-in state
            if data.state.lower() == current_user.state_name.lower():
                daily_cases_by_date[date_str]['state_cases'] += data.positiveIncrease
                daily_cases_by_date[date_str]['state_deaths'] += data.deathIncrease
        
        national_daily_cases = [
            {
                'date': date,
                'usa_cases': data['usa_cases'],
                'usa_deaths': data['usa_deaths'],
                'state_cases': data['state_cases'],
                'state_deaths': data['state_deaths']
            }
            for date, data in sorted(daily_cases_by_date.items())
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
            
            # Calculate trend for this state
            fourteen_days_ago = data.date - timedelta(days=14)
            old_data = PandemicData.query.filter(
                PandemicData.state == data.state,
                PandemicData.date >= fourteen_days_ago,
                PandemicData.date < data.date
            ).order_by(PandemicData.date.asc()).first()
            
            if old_data:
                case_change = ((data.positive - old_data.positive) / old_data.positive) * 100
                if case_change > 20:
                    state_trend = "Increasing"
                elif case_change < -20:
                    state_trend = "Decreasing"
                else:
                    state_trend = "Stable"
            else:
                state_trend = "Unknown"
            
            # Estimate active cases (simplified)
            active_cases_state = int(data.positive * 0.15)
            
            # Determine if state should be confined
            # A state is considered confined if:
            # 1. Risk level is Critical OR
            # 2. Positive rate > 15% OR
            # 3. 14-day trend is increasing with high positive rate
            is_confined = (
                risk_level == "Critical" or
                positive_rate > 0.15 or
                (state_trend == "Increasing" and positive_rate > 0.1)
            )
            
            covid_data[data.state] = {
                'total_cases': data.positive,
                'active_cases': active_cases_state,
                'deaths': data.death,
                'risk_level': risk_level,
                'abbr': get_state_abbreviation(data.state),
                'is_confined': is_confined,
                'trend': state_trend
            }
        
        # Create a class-like object for national data
        class NationalData:
            def __init__(self, total_cases, active_cases, deaths, recovered, state_data=None):
                self.total_cases = total_cases
                self.active_cases = active_cases
                self.deaths = deaths
                self.recovered = recovered
                self.state_total_cases = state_data['total_cases'] if state_data else 0
                self.state_active_cases = state_data['active_cases'] if state_data else 0
                self.state_deaths = state_data['deaths'] if state_data else 0
                self.state_recovered = state_data['recovered'] if state_data else 0
        
        national_data_obj = NationalData(
            total_cases=total_cases,
            active_cases=active_cases,
            deaths=total_deaths,
            recovered=recovered,
            state_data={
                'total_cases': state_total_cases,
                'active_cases': state_active_cases,
                'deaths': state_deaths,
                'recovered': state_recovered
            }
        )
        
        return render_template(
            'main/dashboard.html', 
            covid_data=covid_data,
            national_data=national_data_obj,
            national_daily_cases=national_daily_cases,
            latest_prediction=latest_prediction
        )
    
    # If no data is available
    return render_template('main/dashboard.html', covid_data={}, national_daily_cases=[])

@bp.route('/decision-support')
@login_required
def decision_support():
    # Update predictions before showing decision support
    update_predictions()
    
    # Get the latest data for the state
    latest_data = PandemicData.query.filter(
        PandemicData.state == current_user.state_name
    ).order_by(PandemicData.date.desc()).first()
    
    if not latest_data:
        flash('No data available for your state.', 'error')
        return redirect(url_for('main.dashboard'))
    
    # Get latest prediction for the user's state
    latest_prediction = Prediction.query.filter_by(
        state=current_user.state_name
    ).order_by(desc(Prediction.date)).first()
    
    # Calculate positive rate
    positive_rate = latest_data.positive / latest_data.totalTestResults if latest_data.totalTestResults > 0 else 0
    
    # Determine risk level
    if positive_rate < 0.05:
        risk_level = "Low"
    elif positive_rate < 0.1:
        risk_level = "Medium"
    elif positive_rate < 0.15:
        risk_level = "High"
    else:
        risk_level = "Critical"
    
    # Calculate 14-day trend
    fourteen_days_ago = latest_data.date - timedelta(days=14)
    old_data = PandemicData.query.filter(
        PandemicData.state == current_user.state_name,
        PandemicData.date >= fourteen_days_ago,
        PandemicData.date < latest_data.date
    ).order_by(PandemicData.date.asc()).first()
    
    if old_data:
        case_change = ((latest_data.positive - old_data.positive) / old_data.positive) * 100
        if case_change > 20:
            trend = "Increasing"
        elif case_change < -20:
            trend = "Decreasing"
        else:
            trend = "Stable"
    else:
        trend = "Unknown"
    
    # Generate recommendations based on metrics
    recommendations = []
    
    # High priority recommendations
    if risk_level == "Critical":
        recommendations.append({
            'priority': 'high',
            'title': 'Immediate Action Required',
            'description': 'Your state is experiencing critical levels of COVID-19 spread.',
            'actions': [
                'Implement strict social distancing measures',
                'Consider temporary business closures',
                'Increase testing capacity',
                'Prepare healthcare system for surge'
            ]
        })
    elif risk_level == "High":
        recommendations.append({
            'priority': 'high',
            'title': 'Urgent Measures Needed',
            'description': 'COVID-19 spread is at high levels and requires immediate attention.',
            'actions': [
                'Enforce mask mandates',
                'Limit indoor gatherings',
                'Increase contact tracing efforts',
                'Monitor healthcare capacity'
            ]
        })
    
    # Medium priority recommendations
    if trend == "Increasing":
        recommendations.append({
            'priority': 'medium',
            'title': 'Growing Spread Detected',
            'description': 'Case numbers are trending upward, requiring preventive measures.',
            'actions': [
                'Review and strengthen existing measures',
                'Increase public awareness campaigns',
                'Prepare for potential escalation',
                'Monitor high-risk areas'
            ]
        })
    
    # General recommendations based on metrics
    if positive_rate > 0.1:
        recommendations.append({
            'priority': 'medium',
            'title': 'Testing Strategy Adjustment',
            'description': 'High positive rate indicates need for expanded testing.',
            'actions': [
                'Increase testing availability',
                'Target high-risk communities',
                'Implement regular testing programs',
                'Improve test result turnaround time'
            ]
        })
    
    # Low priority recommendations
    recommendations.append({
        'priority': 'low',
        'title': 'Ongoing Monitoring',
        'description': 'Regular assessment and data collection are essential.',
        'actions': [
            'Maintain daily data collection',
            'Monitor key metrics',
            'Update public communications',
            'Review and adjust measures as needed'
        ]
    })
    
    return render_template(
        'main/decision_support.html',
        risk_level=risk_level,
        positive_rate=positive_rate,
        trend=trend,
        recommendations=recommendations,
        latest_prediction=latest_prediction
    )