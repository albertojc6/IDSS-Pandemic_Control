from flask import Flask, redirect, url_for
from config import Config
from app.extensions import db, login_manager, bcrypt

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    
    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.stats import bp as stats_bp
    app.register_blueprint(stats_bp, url_prefix='/stats')
    
    # Import models after extensions are initialized
    from app import models
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Add a root route for debugging
    @app.route('/')
    def index():
        return redirect(url_for('main.dashboard'))
    
    # Register CLI commands
    from app.cli import register_commands
    register_commands(app)
    
    return app