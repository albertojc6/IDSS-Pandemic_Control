import click
from flask.cli import with_appcontext
from app.extensions import db
from app.models import User
from app.utils.data_generator import generate_sample_data

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    db.create_all()
    click.echo('Initialized the database.')

@click.command('create-admin')
@click.option('--username', default='admin', help='Admin username')
@click.option('--password', default='password', help='Admin password')
@click.option('--email', default='admin@example.com', help='Admin email')
@click.option('--state', default='New York', help='Admin state')
@with_appcontext
def create_admin_command(username, password, email, state):
    """Create an admin user."""
    user = User.query.filter_by(username=username).first()
    if user is None:
        user = User(username=username, email=email, state_name=state)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        click.echo(f'Created admin user: {username} for state: {state}')
    else:
        click.echo(f'Admin user {username} already exists')

@click.command('generate-data')
@click.option('--days', default=30, help='Number of days to generate data for')
@with_appcontext
def generate_data_command(days):
    """Generate sample pandemic data."""
    result = generate_sample_data(days)
    click.echo(result)

def register_commands(app):
    app.cli.add_command(init_db_command)
    app.cli.add_command(create_admin_command)
    app.cli.add_command(generate_data_command)