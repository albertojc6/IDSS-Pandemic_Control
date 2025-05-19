from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from app.models import PandemicData, Prediction, Recommendation
from app.utils.data_generator import get_state_abbreviation
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from collections import defaultdict
from app.main import bp  # Changed to import bp from main blueprint
from app.extensions import db
from app.services.prophet_predictor import ProphetPredictor
from app import check_and_retrain_models
from app.services.fuzzy_epidemiology import FuzzyEpidemiology

@bp.route('/')  # Changed from main to bp
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

def update_predictions(state=None):
    """
    Update predictions for states.
    If state is provided, only update predictions for that state.
    If state is None, update predictions for all states.
    
    Args:
        state: Optional state code to update predictions for. If None, updates all states.
    """
    predictor = current_app.prophet_predictor
    
    # Get list of states to update
    if state:
        states = [state]
    else:
        states = [state[0] for state in PandemicData.query.with_entities(PandemicData.state).distinct().all()]
    
    # Make predictions for each state
    for state in states:
        try:
            # Get the latest date for this specific state
            latest_state_date = PandemicData.query.filter_by(
                state=state
            ).order_by(desc(PandemicData.date)).first()
            
            if not latest_state_date:
                current_app.logger.warning(f"No data found for state {state}")
                continue
                
            latest_date = latest_state_date.date
            latest_datetime = datetime.combine(latest_date, datetime.min.time())
            
            # Get the latest data for this state
            latest_state_data = PandemicData.query.filter_by(
                state=state,
                date=latest_date
            ).first()
            
            if latest_state_data:
                print(f"\nMaking predictions for {state} using data from {latest_date}")
                print("=" * 80)
                print(f"Date: {latest_state_data.date}")
                print(f"Positive Cases: {latest_state_data.positive}")
                print(f"Positive Increase: {latest_state_data.positiveIncrease}")
                print(f"Hospitalized Increase: {latest_state_data.hospitalizedIncrease}")
                print(f"Death Increase: {latest_state_data.deathIncrease}")
                print(f"Total Tests: {latest_state_data.totalTestResults}")
                print("=" * 80)
            
            # Check if models need to be retrained
            check_and_retrain_models(current_app, state)
            
            # Delete existing prediction for this state and date if it exists
            existing_prediction = Prediction.query.filter_by(
                state=state,
                date=latest_date
            ).first()
            if existing_prediction:
                db.session.delete(existing_prediction)
                db.session.commit()
            
            # Create new prediction
            prediction = predictor.predict_for_state(state, latest_datetime)
            db.session.add(prediction)
            db.session.commit()
            current_app.logger.info(f"Updated prediction for {state} using data from {latest_date}")
            
        except Exception as e:
            current_app.logger.error(f"Error making prediction for {state}: {str(e)}")
            if state == current_user.state_name:
                print(f"Error making prediction for {state}: {str(e)}")

@bp.route('/dashboard')
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
        
        # Get latest recommendations for all states
        latest_recommendations = {}
        for state in [data.state for data in latest_data]:
            recommendation = Recommendation.query.filter_by(
                state=state
            ).order_by(desc(Recommendation.date)).first()
            
            if recommendation:
                latest_recommendations[state] = recommendation
        
        # Prepare data for the map using recommendations
        covid_data = {}
        for data in latest_data:
            recommendation = latest_recommendations.get(data.state)
            
            if recommendation:
                # Determine risk level based on recommendation's risk_level
                risk_level = "Low"
                if recommendation.risk_level >= 50:
                    risk_level = "Critical"
                elif recommendation.risk_level >= 30:
                    risk_level = "High"
                elif recommendation.risk_level > 25:
                    risk_level = "Medium"
                
                # Determine if state is confined based on confinement_level
                is_confined = recommendation.confinement_level in ['Selective', 'Strict', 'Immediate']
                
                covid_data[data.state] = {
                    'total_cases': data.positive,
                    'active_cases': int(data.positive * 0.15),  # Estimate active cases
                    'deaths': data.death,
                    'risk_level': risk_level,
                    'abbr': get_state_abbreviation(data.state),
                    'is_confined': is_confined,
                    'trend': "Stable"  # Default trend
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
    
    # Get latest recommendation for the user's state
    latest_recommendation = Recommendation.query.filter_by(
        state=current_user.state_name
    ).order_by(desc(Recommendation.date)).first()
    
    # Calculate positive rate
    positive_rate = latest_data.positive / latest_data.totalTestResults if latest_data.totalTestResults > 0 else 0
    
    # Use fuzzy system risk level if available
    if latest_recommendation:
        risk_level = latest_recommendation.risk_level
        confinement_level = latest_recommendation.confinement_level
        beds_recommendation = latest_recommendation.beds_recommendation
        
        # Obtenir el percentatge de vacunació més recent
        fuzzy_system = FuzzyEpidemiology()
        vaccination_percentages = fuzzy_system._recalculate_vaccination_percentages(latest_data.date)
        vaccination_percentage = vaccination_percentages.get(current_user.state_name, 0.0)
    else:
        # Fallback to basic risk level calculation
        if positive_rate < 0.05:
            risk_level = "Low"
        elif positive_rate < 0.1:
            risk_level = "Medium"
        elif positive_rate < 0.15:
            risk_level = "High"
        else:
            risk_level = "Critical"
        confinement_level = "No"
        beds_recommendation = "Not needed"
        vaccination_percentage = 0.0
    
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
    
    # High priority recommendations based on confinement level
    if confinement_level == "Immediate":
        recommendations.append({
            'priority': 'high',
            'title': 'Immediate Lockdown Required',
            'description': 'Your state is experiencing critical levels of COVID-19 spread requiring immediate and strict measures.',
            'actions': [
                'Implement complete lockdown of non-essential activities',
                'Close all non-essential businesses and services',
                'Restrict movement to essential purposes only',
                'Enforce strict curfew measures',
                'Mobilize additional healthcare resources',
                'Prepare emergency medical facilities'
            ]
        })
    elif confinement_level == "Strict":
        recommendations.append({
            'priority': 'high',
            'title': 'Strict Confinement Measures',
            'description': 'Severe COVID-19 spread requires strict confinement measures.',
            'actions': [
                'Close all non-essential businesses',
                'Limit gatherings to maximum 2 people',
                'Implement strict travel restrictions',
                'Enforce mandatory mask wearing in all public spaces',
                'Increase healthcare capacity',
                'Implement emergency response protocols'
            ]
        })
    elif confinement_level == "Selective":
        recommendations.append({
            'priority': 'medium',
            'title': 'Selective Confinement Measures',
            'description': 'Moderate to high spread requires selective confinement measures.',
            'actions': [
                'Close high-risk businesses and venues',
                'Limit gatherings to maximum 6 people',
                'Implement regional travel restrictions',
                'Enforce mask mandates in indoor spaces',
                'Increase testing and contact tracing',
                'Prepare for potential escalation'
            ]
        })
    
    # Medium priority recommendations based on trend
    if trend == "Increasing":
        recommendations.append({
            'priority': 'medium',
            'title': 'Growing Spread Detected',
            'description': 'Case numbers are trending upward, requiring preventive measures.',
            'actions': [
                'Review and strengthen existing measures',
                'Increase public awareness campaigns',
                'Prepare for potential escalation',
                'Monitor high-risk areas',
                'Enhance testing capacity',
                'Strengthen contact tracing efforts'
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
                'Improve test result turnaround time',
                'Expand testing locations',
                'Prioritize symptomatic individuals'
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
            'Review and adjust measures as needed',
            'Coordinate with neighboring states',
            'Prepare contingency plans'
        ]
    })
    
    return render_template(
        'main/decision_support.html',
        risk_level=risk_level,
        positive_rate=positive_rate,
        trend=trend,
        recommendations=recommendations,
        latest_prediction=latest_prediction,
        confinement_level=confinement_level,
        beds_recommendation=beds_recommendation,
        vaccination_percentage=vaccination_percentage,
        ia=latest_recommendation.ia,
        theta=latest_recommendation.theta,
        pi=latest_recommendation.pi,
        lethality=latest_recommendation.lethality,
        pop_over_65=latest_recommendation.pop_over_65,
        density=latest_recommendation.density
    )