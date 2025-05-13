import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))
rootdir = os.path.dirname(basedir)  # Go up one level to reach the app root

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Template and static folder configuration
    TEMPLATES_FOLDER = os.path.join(rootdir, 'front-end', 'templates')
    STATIC_FOLDER = os.path.join(rootdir, 'front-end', 'static')
    
    # Add upload folder configuration
    UPLOAD_FOLDER = os.path.join(basedir, 'data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Make sure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)