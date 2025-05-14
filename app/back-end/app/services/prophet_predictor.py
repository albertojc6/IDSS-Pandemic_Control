from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from app.models import PandemicData, Prediction
from app.extensions import db
from sqlalchemy import desc


class ProphetPredictor:
    """
    Class for loading trained Prophet models and making predictions
    """
    def __init__(self, model_dir: str = None):
        """
        Args:
            model_dir: Path to directory containing trained models
        """
        # Set up base paths
        self.base_path = Path(__file__).parent.parent.parent / "data"
        
        if model_dir is None:
            self.model_dir = self.base_path / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.models = {}
        self.target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]

        # Load cluster assignments
        cluster_path = self.base_path / "clustering" / "state_clusters.csv"
        self.cluster_df = pd.read_csv(cluster_path)
        
        # Load static state data for regressors
        self.df_est = pd.read_csv(self.base_path / "preprocessed" / "dataMatrix" / "static_stateMatrix.csv")
        
        # Load daily COVID data
        self.df_tmp = pd.read_csv(self.base_path / "preprocessed" / "dataMatrix" / "daily_covidMatrix.csv")
        self.df_tmp['ds'] = pd.to_datetime(self.df_tmp['date'])
        
    def load_models(self):
        """
        Load all trained Prophet models
        """
        for target in self.target_lst:
            clusters = self.cluster_df[target].unique()
            for clust in clusters:
                model_key = f"{target}_cluster{clust}"
                model_path = self.model_dir / f"{model_key}.pkl"
                
                if model_path.exists():
                    with open(model_path, 'rb') as fin:
                        model = pickle.load(fin)
                    self.models[model_key] = model
                else:
                    print(f"Warning: Model {model_key} not found")
    
    def predict_for_state(self, state: str, date: pd.Timestamp) -> Prediction:
        """
        Make predictions for a specific state and date
        
        Args:
            state: State code
            date: Date to make predictions for
            
        Returns:
            Prediction object with the 7-day sums and daily predictions
        """
        # Get historical data for the state from daily_covidMatrix
        state_data = self.df_tmp[self.df_tmp['state'] == state].copy()
        
        if state_data.empty:
            raise ValueError(f"No data found for state {state}")
        
        # Get static regressors for the state
        state_static = self.df_est[self.df_est['state'] == state]
        if state_static.empty:
            raise ValueError(f"No static data found for state {state}")
        
        # Prepare future data with both temporal and static features
        future = self._prepare_future(state, date, state_data, state_static)
        
        # Get cluster assignments
        cluster = self.cluster_df.loc[self.cluster_df['state'] == state]
        if cluster.empty:
            raise ValueError(f"No cluster assignment found for state {state}")
        
        # Make predictions for each target
        predictions = {}
        daily_predictions = {}
        for target in self.target_lst:
            cluster_num = cluster[target].values[0]
            model_key = f"{target}_cluster{cluster_num}"
            
            if model_key not in self.models:
                raise ValueError(f"No model found for {target} in cluster {cluster_num}")
            
            forecast = self.models[model_key].predict(future)
            daily_predictions[target] = [int(x) for x in forecast['yhat']]
            predictions[target] = sum(daily_predictions[target])
        
        # Create and save prediction record
        prediction = Prediction(
            state=state,
            date=date.date(),
            positive_increase_sum=predictions['positiveIncrease'],
            hospitalized_increase_sum=predictions['hospitalizedIncrease'],
            death_increase_sum=predictions['deathIncrease'],
            positive_daily=daily_predictions['positiveIncrease'],
            hospitalized_daily=daily_predictions['hospitalizedIncrease'],
            death_daily=daily_predictions['deathIncrease']
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        return prediction
    
    def _prepare_future(self, state: str, start_date: pd.Timestamp, df_train: pd.DataFrame, state_static: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to prepare future data for predictions
        
        Args:
            state: State to predict for
            start_date: Start date for predictions
            df_train: DataFrame containing training data
            state_static: DataFrame containing static state features
        """
        # Get temporal features from training data
        temporal_features = [col for col in df_train.columns if col not in ['ds', 'state', 'date'] + self.target_lst]
        last_temporal = df_train[temporal_features].iloc[-1].values
        
        # Get static features
        static_features = [col for col in state_static.columns if col != 'state']
        static_values = state_static[static_features].values[0]
        
        # Combine all features
        all_features = temporal_features + static_features
        all_values = np.concatenate([last_temporal, static_values])
        
        # Create future dates
        future_dates = pd.date_range(
            start=start_date,
            periods=7,
            freq='D'
        )
        
        # Create future DataFrame with all features
        future = pd.DataFrame({'ds': future_dates})
        future_reg = pd.DataFrame([all_values]*7, columns=all_features)
        
        return pd.concat([future, future_reg], axis=1)