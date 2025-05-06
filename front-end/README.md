# IDSS Pandemic Management System

## Setup and Running the Application

1. First, ensure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Initialize the database:
```bash
flask init-db
```

3. Create an admin user (optional, defaults shown):
```bash
flask create-admin --username admin --password password --email admin@example.com --state "New York"
```

4. Generate sample pandemic data (optional, defaults to 30 days):
```bash
flask generate-data --days 30
```

5. Run the application:
```bash
python run.py
```

The application will be available at `http://localhost:5000`

## CLI Commands

The application provides several CLI commands for setup and maintenance:

- `flask init-db`: Creates the database tables
- `flask create-admin`: Creates an admin user with specified credentials
  - Options:
    - `--username`: Admin username (default: 'admin')
    - `--password`: Admin password (default: 'password')
    - `--email`: Admin email (default: 'admin@example.com')
    - `--state`: Admin state (default: 'New York')
- `flask generate-data`: Generates sample pandemic data
  - Options:
    - `--days`: Number of days to generate data for (default: 30)

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
