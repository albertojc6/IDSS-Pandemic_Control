from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from itertools import product
from typing import Dict, List, Tuple

# =========================
# ProphetTrainer: Training and Evaluation of Prophet Models for US States
# =========================
# This script provides a class to:
#   - Prepare and split COVID-19 data for US states
#   - Cluster states based on temporal and structural similarity
#   - Train Prophet models for each cluster and target variable
#   - Save/load models using pickle
#   - Evaluate model performance on a test set
# =========================

def prepare_data():
    """
    Loads, merges, and splits the COVID-19 data for all US states.
    Returns:
        df_train: DataFrame for training (before 2021-03-01)
        df_test: DataFrame for testing (from 2021-03-01)
    """
    def load_data():
        """
        Loads the preprocessed daily and static datasets.
        Returns:
            df_tmp: Daily COVID-19 data
            df_est: Static state-level data
        """
        # Use script's location to resolve data path
        base_path = Path(__file__).parent.parent.parent / "data/preprocessed/dataMatrix"
        print(f"Loading data from: {base_path}")
        df_tmp = pd.read_csv(base_path / "daily_covidMatrix.csv") 
        df_est = pd.read_csv(base_path / "static_stateMatrix.csv")
        return df_tmp, df_est
    
    # Load the necessary data
    df_tmp, df_est = load_data()

    # Target variables to predict
    target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]

    # Prepare temporal data
    df_tmp = df_tmp.rename(columns={'date': 'ds'})
    df_tmp = df_tmp[["ds", "state"] + target_lst]

    # Merge temporal and static data
    df = pd.merge(df_tmp, df_est, on='state', how='left')

    def split_data(df, date_col = "ds"):
        """
        Splits the data into train and test sets by date.
        Returns:
            df_train: DataFrame for training
            df_test: DataFrame for testing
        """
        df[date_col] = pd.to_datetime(df[date_col])
        df_train = df[df[date_col] < '2021-03-01']
        df_test = df[df[date_col] >= '2021-03-01']
        return df_train, df_test
    
    # Split the data
    df_train, df_test = split_data(df)
    return df_train, df_test


class ProphetTrainer:
    """
    Trains and evaluates Prophet models for all US states, using clustering to group similar states.
    """
    def __init__(self,
                 w_temp: float = 0.3,
                 t_clust: float = 1,
                 verbose: int = 0
        ) -> None:
        """
        Initializes the trainer, loads data, and sets up directories.
        Args:
            w_temp: Weight for temporal component in clustering distances
            t_clust: Distance threshold for clustering
            verbose: If >0, show plots interactively
        """
        assert (w_temp <= 1), "temporal weight must be less or equal to 1!"
        self.w_temp = w_temp
        self.t_clust = t_clust
        self.verbose = verbose

        # Load data for training and testing
        self.df_train, self.df_test = prepare_data()
        self.states = self.df_train['state'].unique()

        # Target variables to predict
        self.target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]
        # Additional regressors (static features)
        self.est_regressors = self.df_train.drop(columns=(['ds', 'state'] + self.target_lst)).columns.tolist()

        # Directory to store models
        self.model_dir = Path(__file__).parent.parent / "src_models/models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # In-memory model storage
        self.models = {}
        
        # Define shared parameter grid for all models
        self.param_grid = {
            "changepoint_prior_scale": [0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 1, 10.0]
        }

    def cluster_states(self):
        """
        Clusters states based on a weighted combination of temporal and structural distances.
        Saves cluster assignments to CSV and dendrograms to PNG.
        Returns:
            cluster_df: DataFrame with state-to-cluster assignments for each target
        """
        def load_data():
            """
            Loads precomputed distance matrices for temporal and structural data.
            Returns:
                df_tmp_DM: Dict of temporal distance matrices (per target)
                df_est_DM: Structural distance matrix
            """
            base_path = Path(__file__).parent.parent / "data/preprocessed/dataMatrix/state_distances"
            df_tmp_DM = {
                'positiveIncrease': pd.read_csv(base_path / "tmp_dist_Pos.csv"), 
                'hospitalizedIncrease': pd.read_csv(base_path / "tmp_dist_Hosp.csv"), 
                'deathIncrease': pd.read_csv(base_path / "tmp_dist_Death.csv")
            }
            df_est_DM = pd.read_csv(base_path / "est_dist.csv")
            return df_tmp_DM, df_est_DM
        
        # Load distance matrices
        df_tmp_DM, df_est_DM = load_data()

        def _toMatrix(df):
            """
            Converts a DataFrame to a numpy distance matrix (excluding first column).
            """
            return df.iloc[:, 1:].values

        def agg_clustering(temp_comp_DM):
            """
            Performs hierarchical clustering using a weighted sum of temporal and structural distances.
            Args:
                temp_comp_DM: Distance matrix for a temporal component
            Returns:
                Z: Linkage matrix
                clusters: Cluster assignments
            """
            # Normalize both matrices to [0, 1]
            scaler = MinMaxScaler()
            d_temp_normalized = scaler.fit_transform(_toMatrix(temp_comp_DM))
            d_est_normalized = scaler.fit_transform(_toMatrix(df_est_DM))

            # Weighted sum of distances
            dist_matrix_agg = self.w_temp * d_temp_normalized + (1 - self.w_temp) * d_est_normalized

            # Hierarchical clustering (Ward's method)
            Z = linkage(dist_matrix_agg, method='ward')

            # Assign clusters
            clusters = fcluster(Z, t=self.t_clust, criterion='distance')

            # Cophenetic correlation (quality of clustering)
            coph_dist = cophenet(Z)
            original_dist = pdist(dist_matrix_agg)
            correlation = np.corrcoef(original_dist, coph_dist)[0, 1]
            print(f"Cophenetic Correlation: {correlation:.3f}")
            return Z, clusters
            
        def show_clustering(Z, c_name):
            """
            Saves or shows a dendrogram for the clustering result.
            Args:
                Z: Linkage matrix
                c_name: Name of the target variable
            """
            plt.figure(figsize=(12, 6))
            dendrogram(Z, labels=self.states, color_threshold=self.t_clust)
            plt.axhline(y=self.t_clust, color='b', linestyle='--', label=f'Cut at distance={self.t_clust}')
            plt.legend()
            plt.title(f"Hierarchical Clustering Dendrogram ({c_name})")
            plt.xlabel("States")
            plt.ylabel("Distance")
            if self.verbose:    
                plt.show()
            else:
                base_path = Path(__file__).parent.parent / "src_models/clustering"
                plt.savefig(f"{base_path / f'clust_{c_name}.png'}", dpi=300, bbox_inches='tight')
                
        # Perform clustering for each target variable
        cluster_comp = dict()
        for c_name, c_DM in df_tmp_DM.items():
            Z, clusters = agg_clustering(c_DM)
            show_clustering(Z, c_name)
            cluster_comp[c_name] = pd.DataFrame({
                "state": self.states,
                c_name: clusters
            })

        # Print number of clusters for each target
        for c_name, clust_df in cluster_comp.items():
            print(f"Nº of clusters for {c_name}: {len(clust_df[c_name].unique())}")

        # Merge cluster assignments into a single DataFrame
        self.cluster_df = cluster_comp['positiveIncrease'].merge(cluster_comp['hospitalizedIncrease'], on='state') \
                .merge(cluster_comp['deathIncrease'], on='state')
        
        # Save cluster assignments
        base_path = Path(__file__).parent.parent / "src_models/clustering"
        self.cluster_df.to_csv(f"{base_path / 'state_clusters.csv'}", index=False)
        return self.cluster_df

    def _grid_search(self, df_train: pd.DataFrame, target: str, cluster_states: List[str]) -> Tuple[Dict, float]:
        """
        Performs grid search to find optimal Prophet parameters for a specific model (cluster).
        
        Args:
            df_train: Training DataFrame
            target: Target variable name
            cluster_states: List of states in this cluster
            
        Returns:
            Tuple of (best_params, best_score)
        """
        best_score = float('inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(self.param_grid.keys(), v)) for v in product(*self.param_grid.values())]
        
        # Filter data for this cluster
        df_cluster = df_train[df_train['state'].isin(cluster_states)]
        
        # Use last 7 days for validation
        validation_start = df_cluster['ds'].max() - pd.Timedelta(days=7)
        df_val = df_cluster[df_cluster['ds'] >= validation_start]
        df_train_subset = df_cluster[df_cluster['ds'] < validation_start]
        
        # Base Prophet parameters that work well for COVID-19 data
        base_params = {
            "growth": "linear",
            "changepoint_range": 0.8,
            "yearly_seasonality": False,
            "weekly_seasonality": True,  # pandemic effects are weekly
            "daily_seasonality": False,
            "seasonality_mode": "additive"
        }
        
        for params in param_combinations:
            try:
                # Combine base parameters with grid search parameters
                model_params = {**base_params, **params}
                
                # Create and configure model
                model = Prophet(**model_params)
                for reg in self.est_regressors:
                    model.add_regressor(reg)
                
                # Fit on training subset with error handling
                df_prophet = df_train_subset[['ds', target] + self.est_regressors].rename(columns={target: 'y'})
                
                try:
                    model.fit(df_prophet)
                except Exception as e:
                    print(f"Fit error with parameters {params}: {str(e)}")
                    continue
                
                # Make predictions for validation period
                future = pd.DataFrame({'ds': df_val['ds']})
                for reg in self.est_regressors:
                    future[reg] = df_val[reg].values
                
                try:
                    forecast = model.predict(future)
                except Exception as e:
                    print(f"Prediction error with parameters {params}: {str(e)}")
                    continue
                
                # Calculate validation score (MAE) for each day
                y_true = df_val[target].values
                y_pred = forecast['yhat'].values
                
                # Calculate MAE for each day
                score = mean_absolute_error(y_true, y_pred)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    print(f"New best parameters found: {params} (score: {score:.2f})")
                    # Print daily predictions vs actual for best parameters
                    print("\nDaily predictions vs actual:")
                    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                        print(f"Day {i+1}: Predicted = {pred:.2f}, Actual = {true}")
                    
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        if best_params is None:
            print(f"Warning: No valid parameters found for {target}_cluster{len(cluster_states)}. Using default parameters.")
            best_params = {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0
            }
            best_score = float('inf')
        
        return best_params, best_score

    def train_prophet(self):
        """
        Trains Prophet models for each cluster and target variable using optimized parameters.
        Models are saved to disk and also kept in memory for evaluation.
        """
        cluster_df = self.cluster_df
        
        for target in self.target_lst:
            clusters = cluster_df[target].unique()
            for clust in clusters:
                model_key = f"{target}_cluster{clust}"
                model_path = self.model_dir / f"{model_key}.pkl"
                
                # If model already exists, load it
                if model_path.exists():
                    print(f"Loading existing model: {model_key}")
                    with open(model_path, 'rb') as fin:
                        model = pickle.load(fin)
                    self.models[model_key] = model
                    continue
                
                # Get states in this cluster
                cluster_states = cluster_df[cluster_df[target] == clust]['state'].unique()
                
                # Find optimal parameters for this specific model
                print(f"\nOptimizing parameters for {model_key}...")
                best_params, best_score = self._grid_search(self.df_train, target, cluster_states)
                print(f"Best parameters for {model_key}: {best_params}")
                print(f"Best validation score: {best_score:.2f}")
                
                # Train model with optimized parameters
                df_clust = self.df_train[self.df_train['state'].isin(cluster_states)]
                df_prophet = df_clust[['ds', target] + self.est_regressors].rename(columns={target: 'y'})
                
                # Combine base parameters with best parameters
                model_params = {
                    "growth": "linear",
                    "changepoint_range": 0.8,
                    "yearly_seasonality": False,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "seasonality_mode": "additive",
                    **best_params
                }
                
                model = Prophet(**model_params)
                
                for reg in self.est_regressors:
                    model.add_regressor(reg)
                
                try:
                    model.fit(df_prophet)
                    print(f"Trained model for {model_key}")
                    
                    with open(model_path, 'wb') as fout:
                        pickle.dump(model, fout)
                    
                    self.models[model_key] = model
                except Exception as e:
                    print(f"Error training final model for {model_key}: {str(e)}")
                    continue

    def evaluate_models(self):
        """
        Evaluates all trained models on the test set (first 7 days for each state).
        Prints and returns MAE, RMSE, and MAPE for each state and target.
        Returns:
            metrics: List of dicts with evaluation results
        """
        metrics = []
        for state in self.states:
            state_test = self.df_test[self.df_test['state'] == state]
            if state_test.empty:
                continue
            # Use first 7 days of test set
            test_dates = state_test['ds'].unique()[:7]
            df_test_7d = state_test[state_test['ds'].isin(test_dates)]
            for target in self.target_lst:
                cluster = self.cluster_df.loc[self.cluster_df['state'] == state, target].values[0]
                model_key = f"{target}_cluster{cluster}"
                if model_key not in self.models:
                    continue
                # Prepare future dataframe for prediction
                future = self._prepare_future(state, test_dates[0])
                forecast = self.models[model_key].predict(future)
                
                # Get daily predictions and actual values
                y_true = df_test_7d[target].values
                y_pred = np.maximum(0, forecast['yhat'].values)  # Clip negative predictions to 0
                
                # Calculate metrics for each day
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else 0.0

                metrics.append({
                    'target': target,
                    'state': state,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
                
                # Print daily predictions vs actual
                print(f"\n{target} - {state} predictions:")
                print("Day\tPredicted\tActual")
                print("-" * 30)
                for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                    print(f"{i+1}\t{pred:.2f}\t\t{true}")
                print(f"MAE: {mae:.2f}")
                print(f"RMSE: {rmse:.2f}")
                print(f"MAPE: {mape:.2f}%")
                print("-" * 30)
        
        return metrics

    def _prepare_future(self, state: str, start_date: pd.Timestamp) -> pd.DataFrame:
        """
        Prepares a DataFrame for future prediction (7 days) for a given state.
        Args:
            state: State to predict for
            start_date: Start date for prediction
        Returns:
            DataFrame with future dates and static regressors
        """
        state_train = self.df_train[self.df_train['state'] == state]
        last_regressors = state_train[self.est_regressors].iloc[-1].values
        future_dates = pd.date_range(
            start=start_date,
            periods=7,
            freq='D'
        )
        future = pd.DataFrame({'ds': future_dates})
        future_reg = pd.DataFrame([last_regressors]*7, columns=self.est_regressors)
        return pd.concat([future, future_reg], axis=1)


if __name__ == "__main__":
    # Example usage: cluster, train, and evaluate
    trainer = ProphetTrainer()
    trainer.cluster_states()
    trainer.train_prophet()
    trainer.evaluate_models() 