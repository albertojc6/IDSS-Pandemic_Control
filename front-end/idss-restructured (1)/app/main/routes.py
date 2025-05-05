from flask import render_template, url_for
from flask_login import login_required, current_user
from app.main import bp
import urllib.parse

@bp.route('/dashboard')
@login_required
def dashboard():
    # Format state name for the flag URL
    state_flag_name = current_user.state_name.lower().replace(' ', '-')
    return render_template('main/dashboard.html', state_flag_name=state_flag_name)

