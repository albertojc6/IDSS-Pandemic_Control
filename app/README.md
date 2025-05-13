# Pandemic Control Application

This is a web application for pandemic control management with a Flask backend and a frontend interface.

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install backend dependencies:
```bash
cd back-end
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python run.py
```
The backend server will run on http://localhost:5000

### Frontend Setup

The frontend is served directly by the Flask backend from the `front-end` directory. No additional setup is required.

## Running the Application

1. Make sure you have completed the backend setup steps above
2. The backend server will serve both the API endpoints and the frontend static files
3. Access the application by opening your web browser and navigating to:
   http://localhost:5000

## Project Structure

- `back-end/`: Contains the Flask backend application
  - `app/`: Main application code
  - `run.py`: Application entry point
  - `config.py`: Configuration settings
  - `requirements.txt`: Python dependencies

- `front-end/`: Contains the frontend files
  - `static/`: Static assets (CSS, JavaScript, images)
  - `templates/`: HTML templates

## API Endpoints

The backend provides RESTful API endpoints that the frontend can consume. Make sure the backend server is running before accessing the frontend.