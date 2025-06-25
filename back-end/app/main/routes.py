from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from app.models import PandemicData, Prediction, Recommendation, SatisfactionRating, RecommendationCheck
from app.utils.data_generator import get_state_abbreviation
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from collections import defaultdict
from app.main import bp  # Changed to import bp from main blueprint
from app.extensions import db
from app.services.prophet_predictor import ProphetPredictor
from app import check_and_retrain_models
from app.services.fuzzy_epidemiology import FuzzyEpidemiology
from app.services.kpi_submodules import RecommendationTracker, PrecisionTracker, PreventedTracker, SatisfactionTracker, PredictionKPITracker
import pandas as pd
import numpy as np

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
    
    # Get latest recommendation checks for this recommendation
    recommendation_checks = {}
    if latest_recommendation:
        checks = RecommendationCheck.query.filter_by(
            recommendation_id=latest_recommendation.id
        ).all()
        recommendation_checks = {check.recommendation_id: check.was_taken for check in checks}
    
    # Calculate positive rate
    positive_rate = latest_data.positive / latest_data.totalTestResults if latest_data.totalTestResults > 0 else 0
    
    # Use fuzzy system risk level if available
    if latest_recommendation:
        risk_level = latest_recommendation.risk_level
        confinement_level = latest_recommendation.confinement_level
        beds_recommendation = latest_recommendation.beds_recommendation
        
        # Get vaccination percentage
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
            ],
            'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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
            ],
            'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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
            ],
            'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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
            ],
            'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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
            ],
            'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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
        ],
        'was_taken': recommendation_checks.get(latest_recommendation.id, False) if latest_recommendation else False
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

@bp.route('/update-recommendation-checks', methods=['POST'])
@login_required
def update_recommendation_checks():
    # Get the latest recommendation for the user's state
    latest_recommendation = Recommendation.query.filter_by(
        state=current_user.state_name
    ).order_by(desc(Recommendation.date)).first()

    if not latest_recommendation:
        flash('No recommendations available to update.', 'error')
        return redirect(url_for('main.decision_support'))

    # Get the latest date from pandemic data
    latest_date = PandemicData.query.filter_by(
        state=current_user.state_name
    ).order_by(desc(PandemicData.date)).first().date

    # Get the checked recommendations from the form
    checked_recommendations = request.form.getlist('recommendation_checks[]')

    # Delete existing checks for this recommendation
    RecommendationCheck.query.filter_by(
        recommendation_id=latest_recommendation.id
    ).delete()

    # Create new checks
    for recommendation_title in checked_recommendations:
        check = RecommendationCheck(
            date=latest_date,
            state=current_user.state_name,
            recommendation_id=latest_recommendation.id,
            was_taken=True
        )
        db.session.add(check)

    try:
        db.session.commit()
        flash('Recommendation checks updated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error updating recommendation checks.', 'error')
        current_app.logger.error(f"Error updating recommendation checks: {str(e)}")

    return redirect(url_for('main.decision_support'))

@bp.route('/kpi')
@login_required
def kpi_dashboard():
    # Get the latest date from the database
    latest_date = PandemicData.query.with_entities(func.max(PandemicData.date)).scalar()
    thirty_days_ago = latest_date - timedelta(days=30)

    # Debug: Print date range
    print(f"\nKPI Dashboard Date Range:")
    print(f"Latest date: {latest_date}")
    print(f"30 days ago: {thirty_days_ago}")

    # Initialize KPI trackers
    prediction_tracker = PredictionKPITracker()

    # Generate data for the 4 KPIs using the same date range as dashboard
    # 1. Recommendation Taken Ratio (weekly data)
    dates = pd.date_range(start=thirty_days_ago, end=latest_date, freq='W')
    date_labels = list(dates.strftime('%Y-%m-%d'))
    # Ensure the latest date is included as a label
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    if latest_date_str not in date_labels:
        date_labels.append(latest_date_str)

    # Get all recommendations and their checks for the date range
    recommendations = Recommendation.query.filter(
        Recommendation.date >= thirty_days_ago,
        Recommendation.date <= latest_date,
        Recommendation.state == current_user.state_name
    ).all()

    # Create a dictionary to store ratios
    ratios = {}

    for label in date_labels:
        label_date = pd.to_datetime(label).date()
        # For the latest date, use only recommendations for that day
        if label == latest_date_str:
            day_recommendations = [r for r in recommendations if r.date == label_date]
            if day_recommendations:
                recommendation_ids = [r.id for r in day_recommendations]
                checks = RecommendationCheck.query.filter(
                    RecommendationCheck.recommendation_id.in_(recommendation_ids),
                    RecommendationCheck.date == label_date
                ).all()
                total_recommendations = len(day_recommendations)
                taken_recommendations = len([c for c in checks if c.was_taken])
                # Ensure ratio is between 0 and 1
                ratio = min(1.0, taken_recommendations / total_recommendations) if total_recommendations > 0 else 0
            else:
                # Mock data between 0.65 and 0.85 (65% to 85%)
                ratio = round(np.random.uniform(0.65, 0.85), 2)
            ratios[label] = ratio
        else:
            # For other dates, use week logic
            week_start = label_date
            week_end = week_start + timedelta(days=6)
            week_recommendations = [r for r in recommendations if week_start <= r.date <= week_end]
            if week_recommendations:
                recommendation_ids = [r.id for r in week_recommendations]
                checks = RecommendationCheck.query.filter(
                    RecommendationCheck.recommendation_id.in_(recommendation_ids),
                    RecommendationCheck.date >= week_start,
                    RecommendationCheck.date <= week_end
                ).all()
                total_recommendations = len(week_recommendations)
                taken_recommendations = len([c for c in checks if c.was_taken])
                # Ensure ratio is between 0 and 1
                ratio = min(1.0, taken_recommendations / total_recommendations) if total_recommendations > 0 else 0
            else:
                # Mock data between 0.65 and 0.85 (65% to 85%)
                ratio = round(np.random.uniform(0.65, 0.85), 2)
            ratios[label] = ratio

    recommendation_data = {
        'labels': list(ratios.keys()),
        'datasets': [{
            'label': 'Recommendation Taken Ratio',
            'data': list(ratios.values()),
            'borderColor': 'rgba(75, 192, 192, 1)',
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'fill': True,
            'tension': 0.4
        }]
    }

    # 2. Prediction Precision (daily data)
    dates = pd.date_range(start=thirty_days_ago, end=latest_date, freq='D')

    # Get predictions for the user's state
    predictions = Prediction.query.filter(
        Prediction.date >= thirty_days_ago,
        Prediction.date <= latest_date,
        Prediction.state == current_user.state_name
    ).order_by(Prediction.date).all()

    # Create a dictionary of date -> prediction for easy lookup
    prediction_dict = {pred.date: pred.positive_increase_sum for pred in predictions}

    # Generate precision data
    precision_values = []
    for date in dates:
        date_obj = date.date()
        if date_obj in prediction_dict:
            # Use real prediction if available
            prediction = prediction_dict[date_obj]
            error = prediction_tracker.get_prediction_error(prediction)
            precision_values.append(error)
        elif date_obj < latest_date:
            # Use mock prediction for past dates without real data
            mock_prediction = np.random.randint(100, 1000)
            error = prediction_tracker.get_prediction_error(mock_prediction)
            precision_values.append(error)
        else:
            # For future dates, use None to show no data
            precision_values.append(None)

    precision_data = {
        'labels': dates.strftime('%Y-%m-%d').tolist(),
        'datasets': [{
            'label': 'Prediction Error',
            'data': precision_values,
            'borderColor': 'rgba(255, 99, 132, 1)',
            'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            'fill': True,
            'tension': 0.4,
            'spanGaps': True  # This will connect points across null values
        }]
    }

    # 3. Prevented Saturation (daily data)
    dates = pd.date_range(start=thirty_days_ago, end=latest_date, freq='D')

    # Generate prevented saturation data
    prevented_values = []
    for date in dates:
        date_obj = date.date()
        if date_obj in prediction_dict:
            # Use real prediction if available
            prediction = prediction_dict[date_obj]
            prevented = prediction_tracker.get_prevented_saturation(prediction)
            prevented_values.append(prevented)
        elif date_obj < latest_date:
            # Use mock prediction for past dates without real data
            mock_prediction = np.random.randint(100, 1000)
            prevented = prediction_tracker.get_prevented_saturation(mock_prediction)
            prevented_values.append(prevented)
        else:
            # For future dates, use None to show no data
            prevented_values.append(None)

    prevented_data = {
        'labels': dates.strftime('%Y-%m-%d').tolist(),
        'datasets': [{
            'label': 'Prevented Saturation',
            'data': prevented_values,
            'borderColor': 'rgba(54, 162, 235, 1)',
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'fill': True,
            'tension': 0.4,
            'spanGaps': True  # This will connect points across null values
        }]
    }

    # 4. Satisfaction Score (daily data)
    dates = pd.date_range(start=thirty_days_ago, end=latest_date, freq='D')

    # Get real satisfaction data from database for the current state
    satisfaction_ratings = SatisfactionRating.query.filter(
        SatisfactionRating.date >= thirty_days_ago,
        SatisfactionRating.date <= latest_date,
        SatisfactionRating.state == current_user.state_name
    ).order_by(SatisfactionRating.date).all()

    # Create a dictionary of date -> rating for easy lookup
    rating_dict = {rating.date: rating.rating for rating in satisfaction_ratings}

    # Generate satisfaction data
    satisfaction_values = []
    for date in dates:
        date_obj = date.date()
        if date_obj in rating_dict:
            # Use real data if available
            satisfaction_values.append(rating_dict[date_obj])
        elif date_obj < latest_date:
            # Use mock data (4-5) for past dates without real data
            mock_value = round(np.random.uniform(4, 5), 1)
            satisfaction_values.append(mock_value)
        else:
            # For future dates, use None to show no data
            satisfaction_values.append(None)

    satisfaction_data = {
        'labels': dates.strftime('%Y-%m-%d').tolist(),
        'datasets': [{
            'label': 'Satisfaction Score',
            'data': satisfaction_values,
            'borderColor': 'rgba(255, 206, 86, 1)',
            'backgroundColor': 'rgba(255, 206, 86, 0.2)',
            'fill': True,
            'tension': 0.4,
            'spanGaps': True  # This will connect points across null values
        }]
    }

    return render_template('main/kpi_dashboard.html',
                         title='KPI Dashboard',
                         recommendation_data=recommendation_data,
                         precision_data=precision_data,
                         prevented_data=prevented_data,
                         satisfaction_data=satisfaction_data)