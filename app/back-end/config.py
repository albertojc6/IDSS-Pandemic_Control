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