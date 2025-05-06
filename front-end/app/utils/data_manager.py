import csv
import pandas as pd
from datetime import datetime
from app.extensions import db
from app.models import PandemicData

def parse_date(date_str):
    """Parse date string to datetime object"""
    if not date_str:
        return None
        
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        try:
            return datetime.strptime(date_str, '%m/%d/%Y').date()
        except ValueError:
            return None

def import_csv_data(file_path):
    """Import pandemic data from CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert column names to match model fields if needed
        df.columns = [col.strip() for col in df.columns]
        
        # Process each row
        records_added = 0
        for _, row in df.iterrows():
            date_str = row.get('date')
            if not date_str:
                continue
                
            date = parse_date(date_str)
            if not date:
                continue
                
            # Check if record already exists
            existing = PandemicData.query.filter_by(
                date=date,
                state=row.get('state')
            ).first()
            
            if existing:
                continue
                
            # Create new record
            data = PandemicData(
                date=date,
                state=row.get('state'),
                positive=row.get('positive'),
                totalTestResults=row.get('totalTestResults'),
                death=row.get('death'),
                positiveIncrease=row.get('positiveIncrease'),
                negativeIncrease=row.get('negativeIncrease'),
                total=row.get('total'),
                totalTestResultsIncrease=row.get('totalTestResultsIncrease'),
                posNeg=row.get('posNeg'),
                deathIncrease=row.get('deathIncrease'),
                hospitalizedIncrease=row.get('hospitalizedIncrease'),
                Dose1_Total=row.get('Dose1_Total'),
                Dose1_Total_pct=row.get('Dose1_Total_pct'),
                Dose1_65Plus=row.get('Dose1_65Plus'),
                Dose1_65Plus_pct=row.get('Dose1_65Plus_pct'),
                Complete_Total=row.get('Complete_Total'),
                Complete_Total_pct=row.get('Complete_Total_pct'),
                Complete_65Plus=row.get('Complete_65Plus'),
                Complete_65Plus_pct=row.get('Complete_65Plus_pct')
            )
            db.session.add(data)
            records_added += 1
            
        db.session.commit()
        return records_added
    except Exception as e:
        db.session.rollback()
        raise e

def get_pandemic_data(state=None, start_date=None, end_date=None):
    """Retrieve pandemic data with optional filters"""
    query = PandemicData.query
    
    if state:
        query = query.filter(PandemicData.state == state)
    
    if start_date:
        if isinstance(start_date, str):
            start_date = parse_date(start_date)
        if start_date:
            query = query.filter(PandemicData.date >= start_date)
    
    if end_date:
        if isinstance(end_date, str):
            end_date = parse_date(end_date)
        if end_date:
            query = query.filter(PandemicData.date <= end_date)
    
    # Order by date
    query = query.order_by(PandemicData.date)
    
    return query.all()

def get_states_list():
    """Get list of all states in the database"""
    try:
        states = db.session.query(PandemicData.state).distinct().all()
        return [state[0] for state in states]
    except:
        # Return some default states if database is empty
        return ["NY", "CA", "TX", "FL", "IL"]

def get_time_series_data(state, metric):
    """Get time series data for a specific state and metric"""
    try:
        data = PandemicData.query.filter_by(state=state).order_by(PandemicData.date).all()
        
        dates = [entry.date.strftime('%Y-%m-%d') for entry in data]
        values = [getattr(entry, metric, 0) for entry in data]
        
        return {
            'dates': dates,
            'values': values
        }
    except:
        # Return empty data if there's an error
        return {
            'dates': [],
            'values': []
        }