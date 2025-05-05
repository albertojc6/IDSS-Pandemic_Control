from flask_login import LoginManager
from flask_bcrypt import Bcrypt

# Initialize extensions
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
bcrypt = Bcrypt()
