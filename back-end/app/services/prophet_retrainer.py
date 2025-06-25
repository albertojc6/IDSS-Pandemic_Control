from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sqlalchemy import desc
from app.models import PandemicData, StaticStateData
from app.extensions import db
from app.services.prophet_trainer import ProphetTrainer
import pickle
from prophet import Prophet
import os
import time

class ProphetRetrainer:
    """
    Service for retraining Prophet models when new data is added.
    """
    def __init__(self, app=None):
        """
        Initialize the retrainer.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.base_path = Path(__file__).parent.parent.parent / "data"
        self.model_dir = self.base_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cluster assignments
        self.cluster_path = self.base_path / "clustering" / "state_clusters.csv"
        self.cluster_df = pd.read_csv(self.cluster_path)
        
        # Target variables to predict
        self.target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]
        
        # Dictionary to store last training date for each state
        # Initialize with 2021-02-28 for all states
        self.last_training_dates = {}
        initial_date = datetime(2021, 2, 28).date()
        
        # Get all states from PandemicData
        with self.app.app_context():
            states = [state[0] for state in PandemicData.query.with_entities(PandemicData.state).distinct().all()]
            for state in states:
                self.last_training_dates[state] = initial_date

    def should_retrain(self, state: str) -> bool:
        """
        Check if a model should be retrained for a given state.
        A model should be retrained if there is any new data.
        
        Args:
            state: State code to check
            
        Returns:
            bool: True if model should be retrained, False otherwise
        """
        print(f"\n=== Checking if models need retraining for state: {state} ===")
        self.app.logger.info(f"Checking if models need retraining for state: {state}")
        
        # Get the latest date in the database for this state
        latest_data = PandemicData.query.filter_by(
            state=state
        ).order_by(desc(PandemicData.date)).first()
        
        if not latest_data:
            print(f"No data found for state {state}, skipping retraining check")
            self.app.logger.info(f"No data found for state {state}, skipping retraining check")
            return False
            
        # Get the last training date for this state
        last_training_date = self.last_training_dates.get(state, datetime(2021, 2, 28).date())
            
        # Check if there is any new data since last training
        days_since_training = (latest_data.date - last_training_date).days
        print(f"Days since last training for {state}: {days_since_training}")
        print(f"Latest data date: {latest_data.date}, Last model update: {last_training_date}")
        self.app.logger.info(f"Days since last training for {state}: {days_since_training}")
        self.app.logger.info(f"Latest data date: {latest_data.date}, Last model update: {last_training_date}")
        
        # Always retrain if there is any new data
        should_retrain = days_since_training >= 7   # Changed to >= 7 to only retrain when there's new data
        if should_retrain:
            print(f"Retraining needed for {state} - {days_since_training} days of new data")
            self.app.logger.info(f"Retraining needed for {state} - {days_since_training} days of new data")
        else:
            print(f"No retraining needed for {state} - no new data")
            self.app.logger.info(f"No retraining needed for {state} - no new data")
        
        return should_retrain

    def retrain_models(self, state: str = None):
        """
        Retrain Prophet models for a specific state or all states.
        
        Args:
            state: Optional state code to retrain models for. If None, retrains all states.
        """
        with self.app.app_context():
            # Get list of states to retrain
            if state:
                states = [state]
                print(f"\n=== Starting retraining process for state: {state} ===")
                self.app.logger.info(f"Starting retraining process for state: {state}")
            else:
                states = [state[0] for state in PandemicData.query.with_entities(PandemicData.state).distinct().all()]
                print(f"\n=== Starting retraining process for all states: {', '.join(states)} ===")
                self.app.logger.info(f"Starting retraining process for all states: {', '.join(states)}")
            
            # Initialize trainer
            trainer = ProphetTrainer()
            
            # Retrain models for each state
            for state in states:
                if not self.should_retrain(state):
                    print(f"No need to retrain models for {state}")
                    self.app.logger.info(f"No need to retrain models for {state}")
                    continue
                    
                print(f"\nBeginning retraining for {state}")
                self.app.logger.info(f"Beginning retraining for {state}")
                
                # Get cluster assignments for this state
                state_clusters = {}
                for target in self.target_lst:
                    cluster = self.cluster_df.loc[self.cluster_df['state'] == state, target].values[0]
                    state_clusters[target] = cluster
                    print(f"State {state} belongs to cluster {cluster} for target {target}")
                    self.app.logger.info(f"State {state} belongs to cluster {cluster} for target {target}")
                
                # Get the latest data date for this state
                latest_data = PandemicData.query.filter_by(
                    state=state
                ).order_by(desc(PandemicData.date)).first()
                latest_data_date = latest_data.date
                
                # Retrain models for each target and cluster
                for target, cluster in state_clusters.items():
                    model_key = f"{target}_cluster{cluster}"
                    model_path = self.model_dir / f"{model_key}.pkl"
                    print(f"\nProcessing model {model_key}")
                    self.app.logger.info(f"Processing model {model_key}")
                    
                    # Get states in this cluster
                    cluster_states = self.cluster_df[self.cluster_df[target] == cluster]['state'].unique()
                    print(f"States in cluster {cluster}: {', '.join(cluster_states)}")
                    self.app.logger.info(f"States in cluster {cluster}: {', '.join(cluster_states)}")
                    
                    # Get data for all states in this cluster
                    pandemic_data = PandemicData.query.filter(
                        PandemicData.state.in_(cluster_states)
                    ).order_by(PandemicData.date).all()
                    
                    if not pandemic_data:
                        print(f"No data found for cluster {cluster} of {target}")
                        self.app.logger.warning(f"No data found for cluster {cluster} of {target}")
                        continue
                    
                    print(f"Found {len(pandemic_data)} records for cluster {cluster}")
                    self.app.logger.info(f"Found {len(pandemic_data)} records for cluster {cluster}")
                    
                    # Convert to DataFrame
                    df_train = pd.DataFrame([data.to_dict() for data in pandemic_data])
                    df_train['ds'] = pd.to_datetime(df_train['date'])
                    
                    # Get static data
                    static_data = StaticStateData.query.filter(
                        StaticStateData.state.in_(cluster_states)
                    ).all()
                    df_est = pd.DataFrame([data.to_dict() for data in static_data])
                    
                    # Column name mapping from database to model expected names
                    column_mapping = {
                        'pop_0_9': 'pop_0-9',
                        'pop_10_19': 'pop_10-19',
                        'pop_20_29': 'pop_20-29',
                        'pop_30_39': 'pop_30-39',
                        'pop_40_49': 'pop_40-49',
                        'pop_50_59': 'pop_50-59',
                        'pop_60_69': 'pop_60-69',
                        'pop_70_79': 'pop_70-79',
                        'pop_80_plus': 'pop_80+',
                        "Non_metro": "Non-metro"
                    }
                    
                    # Rename columns in static data
                    df_est = df_est.rename(columns=column_mapping)
                    
                    # Merge temporal and static data
                    df_train = pd.merge(df_train, df_est, on='state', how='left')
                    
                    # Print available columns for debugging
                    print(f"Available columns after merge: {df_train.columns.tolist()}")
                    
                    # Train model
                    try:
                        print(f"Starting parameter optimization for {model_key}")
                        self.app.logger.info(f"Starting parameter optimization for {model_key}")
                        # Find optimal parameters
                        best_params, best_score = trainer._grid_search(df_train, target, cluster_states)
                        print(f"Best parameters for {model_key}: {best_params}")
                        print(f"Best validation score: {best_score:.2f}")
                        self.app.logger.info(f"Best parameters for {model_key}: {best_params}")
                        self.app.logger.info(f"Best validation score: {best_score:.2f}")
                        
                        # Train final model with optimized parameters
                        df_prophet = df_train[['ds', target] + trainer.est_regressors].rename(columns={target: 'y'})
                        
                        # Create new model instance with optimized parameters
                        model_params = {
                            "growth": "linear",
                            "changepoint_range": 0.8,
                            "yearly_seasonality": False,
                            "weekly_seasonality": True,
                            "daily_seasonality": False,
                            "seasonality_mode": "additive",
                            **best_params  # Add the optimized parameters
                        }
                        
                        print(f"Creating new model instance for {model_key}")
                        self.app.logger.info(f"Creating new model instance for {model_key}")
                        # Create new model instance instead of using existing one
                        model = Prophet(**model_params)
                        
                        # Add regressors
                        for reg in trainer.est_regressors:
                            model.add_regressor(reg)
                        
                        print(f"Fitting model for {model_key}")
                        self.app.logger.info(f"Fitting model for {model_key}")
                        # Fit the model
                        model.fit(df_prophet)
                        
                        print(f"Saving model {model_key} to {model_path}")
                        self.app.logger.info(f"Saving model {model_key} to {model_path}")
                        # Save model
                        with open(model_path, 'wb') as fout:
                            pickle.dump(model, fout)
                            
                        # Update the last training date for this state
                        self.last_training_dates[state] = latest_data_date
                            
                        print(f"Successfully retrained model {model_key}")
                        self.app.logger.info(f"Successfully retrained model {model_key}")
                        
                    except Exception as e:
                        print(f"Error retraining model {model_key}: {str(e)}")
                        self.app.logger.error(f"Error retraining model {model_key}: {str(e)}")
                        continue
                
                print(f"\nCompleted retraining process for state {state}")
                self.app.logger.info(f"Completed retraining process for state {state}") 