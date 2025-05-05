from flask import render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.stats import bp
from app.stats.forms import CovidStatsForm
from datetime import datetime, date

@bp.route('/covid-stats', methods=['GET', 'POST'])
@login_required
def covid_stats_form():
    form = CovidStatsForm()
    current_date = date.today()
    current_time = datetime.now()
    
    # Check if there's an existing submission for today
    existing_submission = None
    existing_index = None
    
    for i, entry in enumerate(current_user.covid_stats):
        submission_date = entry['report_date'].date() if isinstance(entry['report_date'], datetime) else entry['report_date']
        if submission_date == current_date:
            existing_submission = entry
            existing_index = i
            break
    
    # Pre-fill form with existing data if available
    if request.method == 'GET' and existing_submission:
        form.new_cases.data = existing_submission['new_cases']
        form.total_cases.data = existing_submission['total_cases']
        form.new_deaths.data = existing_submission['new_deaths']
        form.total_deaths.data = existing_submission['total_deaths']
        form.new_vaccinations.data = existing_submission['new_vaccinations']
        form.total_vaccinations.data = existing_submission['total_vaccinations']
        form.current_hospitalizations.data = existing_submission['current_hospitalizations']
        form.icu_patients.data = existing_submission['icu_patients']
    
    if form.validate_on_submit():
        # Create a dictionary to store the form data
        stats = {
            'state': current_user.state_name,
            'report_date': datetime.now(),
            'new_cases': form.new_cases.data,
            'total_cases': form.total_cases.data,
            'new_deaths': form.new_deaths.data,
            'total_deaths': form.total_deaths.data,
            'new_vaccinations': form.new_vaccinations.data,
            'total_vaccinations': form.total_vaccinations.data,
            'current_hospitalizations': form.current_hospitalizations.data,
            'icu_patients': form.icu_patients.data,
            'submitted_at': datetime.now()
        }
        
        # Update or add the stats
        if existing_index is not None:
            current_user.covid_stats[existing_index] = stats
            flash('COVID-19 statistics updated successfully!', 'success')
        else:
            current_user.covid_stats.append(stats)
            flash('COVID-19 statistics submitted successfully!', 'success')
        
        return redirect(url_for('main.dashboard'))
    
    return render_template('stats/covid_stats_form.html', form=form, 
                          current_date=current_date, 
                          current_time=current_time,
                          existing_submission=existing_submission)
