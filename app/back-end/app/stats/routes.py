from flask import render_template, request, jsonify, flash, redirect, url_for, current_app
from app.stats import bp
from app.stats.forms import CovidStatsForm, DailyStatsForm
from app.models import PandemicData
from app.utils.data_manager import import_csv_data, get_pandemic_data, get_states_list, get_time_series_data
from app.extensions import db
import os
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from flask_login import login_required, current_user
from sqlalchemy import desc
from app.main.routes import update_predictions

@bp.route('/')
@login_required
def index():
    return render_template('stats/covid_stats_form.html', form=CovidStatsForm(), title='COVID-19 Statistics')

# Rename this route to covid_stats_form to match the expected endpoint
@bp.route('/covid-stats-form', methods=['GET', 'POST'])
@login_required
def covid_stats_form():
    form = CovidStatsForm()
    if form.validate_on_submit():
        # Process form data
        flash('Statistics request submitted successfully!', 'success')
        return redirect(url_for('stats.covid_stats_form'))
    return render_template('stats/covid_stats_form.html', form=form, title='COVID-19 Statistics')

# Keep the original route for backward compatibility
@bp.route('/covid-stats', methods=['GET', 'POST'])
@login_required
def covid_stats():
    return redirect(url_for('stats.covid_stats_form'))

@bp.route('/data/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                records_added = import_csv_data(file_path)
                flash(f'Successfully imported {records_added} records', 'success')
            except Exception as e:
                flash(f'Error importing data: {str(e)}', 'danger')
                
            return redirect(url_for('stats.data_management'))
            
    return render_template('stats/upload_data.html', title='Upload Data')

@bp.route('/data/management')
@login_required
def data_management():
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Get filter parameters
    state = request.args.get('state')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Query with pagination
    query = PandemicData.query
    
    if state:
        query = query.filter(PandemicData.state == state)
    
    if start_date:
        query = query.filter(PandemicData.date >= start_date)
    
    if end_date:
        query = query.filter(PandemicData.date <= end_date)
    
    pagination = query.order_by(PandemicData.date.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    states = get_states_list()
    
    return render_template(
        'stats/data_management.html',
        pagination=pagination,
        states=states,
        current_state=state,
        start_date=start_date,
        end_date=end_date,
        title='Data Management'
    )

@bp.route('/api/pandemic-data')
@login_required
def api_pandemic_data():
    state = request.args.get('state')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    data = get_pandemic_data(state, start_date, end_date)
    return jsonify([item.to_dict() for item in data])

@bp.route('/api/time-series/<state>/<metric>')
@login_required
def api_time_series(state, metric):
    data = get_time_series_data(state, metric)
    return jsonify(data)

@bp.route('/api/states')
@login_required
def api_states():
    states = get_states_list()
    return jsonify(states)

@bp.route('/daily-stats', methods=['GET', 'POST'])
@login_required
def daily_stats():
    form = DailyStatsForm()
    
    # Get the last day's data for the selected state
    last_day_data = PandemicData.query.filter_by(
        state=current_user.state_name
    ).order_by(desc(PandemicData.date)).first()
    
    # If there's no previous data, use today's date, otherwise use last date + 1 day
    if last_day_data:
        today = last_day_data.date + timedelta(days=1)
    else:
        today = datetime.now().date()
    
    # Check if data already exists for the calculated date
    existing_data = PandemicData.query.filter_by(
        date=today,
        state=current_user.state_name
    ).first()
    
    if existing_data:
        # If data exists, show the existing data instead of the form
        return render_template('stats/daily_stats_view.html', 
                             data=existing_data,
                             title='Today\'s Statistics')
    
    if form.validate_on_submit():
        # Calculate total values based on last day's data and new increases
        positive = (last_day_data.positive + form.positiveIncrease.data) if last_day_data else form.positiveIncrease.data
        totalTestResults = (last_day_data.totalTestResults + form.totalTestResultsIncrease.data) if last_day_data else form.totalTestResultsIncrease.data
        death = (last_day_data.death + form.deathIncrease.data) if last_day_data else form.deathIncrease.data
        total = positive + form.negativeIncrease.data
        posNeg = total
        
        # Create new record
        new_data = PandemicData(
            date=today,
            state=current_user.state_name,
            positive=positive,
            totalTestResults=totalTestResults,
            death=death,
            positiveIncrease=form.positiveIncrease.data,
            negativeIncrease=form.negativeIncrease.data,
            total=total,
            totalTestResultsIncrease=form.totalTestResultsIncrease.data,
            posNeg=posNeg,
            deathIncrease=form.deathIncrease.data,
            hospitalizedIncrease=form.hospitalizedIncrease.data,
            Dose1_Total=form.Dose1_Total.data,
            Dose1_Total_pct=form.Dose1_Total_pct.data,
            Dose1_65Plus=form.Dose1_65Plus.data,
            Dose1_65Plus_pct=form.Dose1_65Plus_pct.data,
            Complete_Total=form.Complete_Total.data,
            Complete_Total_pct=form.Complete_Total_pct.data,
            Complete_65Plus=form.Complete_65Plus.data,
            Complete_65Plus_pct=form.Complete_65Plus_pct.data
        )
        
        try:
            db.session.add(new_data)
            db.session.commit()
            
            # Update predictions after new data is added
            update_predictions()
            
            flash('Daily statistics submitted successfully!', 'success')
            
            # After successful submission, show the submitted data
            return render_template('stats/daily_stats_view.html', 
                                 data=new_data,
                                 title='Today\'s Statistics')
        except Exception as e:
            db.session.rollback()
            flash('Error submitting daily statistics. Please try again.', 'error')
            current_app.logger.error(f'Error submitting daily statistics: {str(e)}')
    
    return render_template('stats/daily_stats_form.html', 
                         form=form, 
                         today=today,
                         state=current_user.state_name,
                         title='Submit Daily Statistics')