from flask import render_template, request, jsonify, flash, redirect, url_for, current_app
from app.stats import bp
from app.stats.forms import CovidStatsForm
from app.models import PandemicData
from app.utils.data_manager import import_csv_data, get_pandemic_data, get_states_list, get_time_series_data
from app.extensions import db
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_login import login_required

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