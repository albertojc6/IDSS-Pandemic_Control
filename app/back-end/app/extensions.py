from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate

# Initialize SQLAlchemy for database operations
# This will be used to define models and perform database queries
db = SQLAlchemy()

# Initialize Flask-Login for user authentication
# Handles user sessions, login/logout functionality, and user loading
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Specify the login view endpoint
login_manager.login_message_category = 'info'  # Set the category for flash messages

# Initialize Bcrypt for password hashing
# Provides secure password hashing and verification
bcrypt = Bcrypt()

# Initialize Flask-Migrate for database migrations
# Handles database schema changes and version control
migrate = Migrate()