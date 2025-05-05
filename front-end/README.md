# IDSS Pandemic Management System

## Running the Application

To run the application:

\`\`\`
python run.py
\`\`\`

## Project Structure

The application follows a blueprint-based structure:

- `app/` - Main application package
  - `__init__.py` - Application factory
  - `extensions.py` - Flask extensions
  - `models.py` - User model and data
  - `auth/` - Authentication blueprint
  - `main/` - Main pages blueprint
  - `stats/` - COVID statistics blueprint
  - `static/` - Static files
    - `flags/` - State flag images
  - `templates/` - HTML templates
    - `auth/` - Authentication templates
    - `main/` - Main page templates
    - `stats/` - Statistics templates
  - `utils/` - Utility scripts
- `config.py` - Configuration settings
- `run.py` - Application entry point
