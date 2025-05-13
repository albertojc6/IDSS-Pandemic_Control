"""
Utility to generate sample pandemic data for visualization
"""
import random
from datetime import datetime, timedelta
from app.models import PandemicData
from app.extensions import db

# US States data with abbreviations
US_STATES = [
    {"name": "Alabama", "abbr": "AL"},
    {"name": "Alaska", "abbr": "AK"},
    {"name": "Arizona", "abbr": "AZ"},
    {"name": "Arkansas", "abbr": "AR"},
    {"name": "California", "abbr": "CA"},
    {"name": "Colorado", "abbr": "CO"},
    {"name": "Connecticut", "abbr": "CT"},
    {"name": "Delaware", "abbr": "DE"},
    {"name": "Florida", "abbr": "FL"},
    {"name": "Georgia", "abbr": "GA"},
    {"name": "Hawaii", "abbr": "HI"},
    {"name": "Idaho", "abbr": "ID"},
    {"name": "Illinois", "abbr": "IL"},
    {"name": "Indiana", "abbr": "IN"},
    {"name": "Iowa", "abbr": "IA"},
    {"name": "Kansas", "abbr": "KS"},
    {"name": "Kentucky", "abbr": "KY"},
    {"name": "Louisiana", "abbr": "LA"},
    {"name": "Maine", "abbr": "ME"},
    {"name": "Maryland", "abbr": "MD"},
    {"name": "Massachusetts", "abbr": "MA"},
    {"name": "Michigan", "abbr": "MI"},
    {"name": "Minnesota", "abbr": "MN"},
    {"name": "Mississippi", "abbr": "MS"},
    {"name": "Missouri", "abbr": "MO"},
    {"name": "Montana", "abbr": "MT"},
    {"name": "Nebraska", "abbr": "NE"},
    {"name": "Nevada", "abbr": "NV"},
    {"name": "New Hampshire", "abbr": "NH"},
    {"name": "New Jersey", "abbr": "NJ"},
    {"name": "New Mexico", "abbr": "NM"},
    {"name": "New York", "abbr": "NY"},
    {"name": "North Carolina", "abbr": "NC"},
    {"name": "North Dakota", "abbr": "ND"},
    {"name": "Ohio", "abbr": "OH"},
    {"name": "Oklahoma", "abbr": "OK"},
    {"name": "Oregon", "abbr": "OR"},
    {"name": "Pennsylvania", "abbr": "PA"},
    {"name": "Rhode Island", "abbr": "RI"},
    {"name": "South Carolina", "abbr": "SC"},
    {"name": "South Dakota", "abbr": "SD"},
    {"name": "Tennessee", "abbr": "TN"},
    {"name": "Texas", "abbr": "TX"},
    {"name": "Utah", "abbr": "UT"},
    {"name": "Vermont", "abbr": "VT"},
    {"name": "Virginia", "abbr": "VA"},
    {"name": "Washington", "abbr": "WA"},
    {"name": "West Virginia", "abbr": "WV"},
    {"name": "Wisconsin", "abbr": "WI"},
    {"name": "Wyoming", "abbr": "WY"},
    {"name": "District of Columbia", "abbr": "DC"}
]

def generate_sample_data(days=30):
    """
    Generate sample pandemic data for all US states for the specified number of days
    """
    # Clear existing data
    db.session.query(PandemicData).delete()
    db.session.commit()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    
    # Generate data for each state
    for state_info in US_STATES:
        state = state_info["name"]
        
        # Base values that will increase over time
        base_positive = random.randint(5000, 50000)
        base_death = random.randint(100, 1000)
        base_total_tests = base_positive * random.randint(5, 10)
        
        current_date = start_date
        while current_date <= end_date:
            # Calculate daily increases (with some randomness)
            positive_increase = random.randint(100, 2000)
            death_increase = random.randint(5, 50)
            test_increase = random.randint(1000, 10000)
            
            # Update base values
            base_positive += positive_increase
            base_death += death_increase
            base_total_tests += test_increase
            
            # Vaccination data
            dose1_total = int(base_positive * random.uniform(1.5, 2.5))
            dose1_total_pct = round(dose1_total / (base_positive * 3) * 100, 1)
            dose1_65plus = int(dose1_total * random.uniform(0.2, 0.4))
            dose1_65plus_pct = round(dose1_65plus / (dose1_total * 0.3) * 100, 1)
            
            complete_total = int(dose1_total * random.uniform(0.7, 0.9))
            complete_total_pct = round(complete_total / (base_positive * 3) * 100, 1)
            complete_65plus = int(complete_total * random.uniform(0.2, 0.4))
            complete_65plus_pct = round(complete_65plus / (complete_total * 0.3) * 100, 1)
            
            # Create pandemic data entry
            pandemic_data = PandemicData(
                date=current_date,
                state=state,
                positive=base_positive,
                totalTestResults=base_total_tests,
                death=base_death,
                positiveIncrease=positive_increase,
                negativeIncrease=test_increase - positive_increase,
                total=base_positive + (test_increase - positive_increase),
                totalTestResultsIncrease=test_increase,
                posNeg=base_positive + (base_total_tests - base_positive),
                deathIncrease=death_increase,
                hospitalizedIncrease=int(positive_increase * random.uniform(0.05, 0.15)),
                Dose1_Total=dose1_total,
                Dose1_Total_pct=dose1_total_pct,
                Dose1_65Plus=dose1_65plus,
                Dose1_65Plus_pct=dose1_65plus_pct,
                Complete_Total=complete_total,
                Complete_Total_pct=complete_total_pct,
                Complete_65Plus=complete_65plus,
                Complete_65Plus_pct=complete_65plus_pct,
                created_at=datetime.now()
            )
            
            db.session.add(pandemic_data)
            current_date += timedelta(days=1)
    
    db.session.commit()
    return f"Generated sample data for {len(US_STATES)} states over {days} days"

def get_state_abbreviation(state_name):
    """Get the abbreviation for a state name"""
    for state in US_STATES:
        if state["name"].lower() == state_name.lower():
            return state["abbr"]
    return None