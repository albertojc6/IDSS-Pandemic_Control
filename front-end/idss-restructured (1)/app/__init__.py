from flask import Flask
from config import Config
from app.extensions import login_manager, bcrypt

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Flask extensions
    login_manager.init_app(app)
    bcrypt.init_app(app)
    
    # Import models to register user_loader
    from app import models
    
    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.stats import bp as stats_bp
    app.register_blueprint(stats_bp, url_prefix='/stats')
    
    @app.route('/')
    def index():
        from flask import redirect, url_for
        return redirect(url_for('auth.login'))
    
    return app
