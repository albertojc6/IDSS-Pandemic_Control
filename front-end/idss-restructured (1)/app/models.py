from flask_login import UserMixin
from datetime import datetime
from app.extensions import login_manager, bcrypt

class User(UserMixin):
    def __init__(self, id, username, password_hash, state_name):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.state_name = state_name
        self.last_login = None
        self.covid_stats = []  # List to store COVID stats submissions

# Dictionary to store users
users = {}

# Generate usernames and passwords for each state
states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", 
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", 
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

# Create users with secure passwords
for i, state in enumerate(states):
    # Create a username based on state abbreviation
    username = state.lower().replace(" ", "")
    
    # Generate a secure password
    password = f"Secure{state.replace(' ', '')}2023!"
    
    # Hash the password
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Create and store the user
    users[username] = User(str(i+1), username, password_hash, state)

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None
