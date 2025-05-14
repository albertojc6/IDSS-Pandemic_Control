from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from prophet.serialize import model_to_json, model_from_json
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from datetime import timedelta
from prophet import Prophet
from pathlib import Path
import pandas as pd
import numpy as np
import os


def prepare_data():
    """
    Returns the merged and splitted data for model training
    """
    def load_data():
        """
        Returns the preprocessed datasets: daily_covidMatrix and static_stateMatrix
        """
        # Resolve based on this script’s location (not the CWD)
        base_path = Path(__file__).parent.parent / "data/preprocessed/dataMatrix"

        df_tmp = pd.read_csv(base_path / "daily_covidMatrix.csv") 
        df_est = pd.read_csv(base_path / "static_stateMatrix.csv")
        
        return df_tmp, df_est
    
    # Load the necessary data
    df_tmp, df_est = load_data()

    # Keep target data from temporal component
    target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]

    df_tmp = df_tmp.rename(columns={'date': 'ds'})
    df_tmp = df_tmp[["ds", "state"] + target_lst]

    # Merge components for complete information
    df = pd.merge(df_tmp, df_est, on='state', how='left')

    def split_data(df, date_col = "ds"):
        """
        Returns the train & test partitions for model training, spliting by date
        """
        # Data goes from 2020/03/07 to 2021/03/07- > 1 year
        df[date_col] = pd.to_datetime(df[date_col])

        df_train = df[df[date_col] < '2021-03-01']
        df_test = df[df[date_col] >= '2021-03-01']

        return df_train, df_test
    
    # Split the data in 2 partitions
    df_train, df_test = split_data(df)

    return df_train, df_test


class ProphetTrainer():
    """
    Encapsulate the training from scratch of all EEUU states Prophet models
    """
    def __init__(self,
                 w_temp: float = 0.3,
                 t_clust: float = 1,
                 verbose: int = 0
        ) -> None:
        """
        Args:
            w_temp: weight for temproal component in clustering distances
            t_clust: distance threshold to get the clusters
        """
        assert (w_temp <= 1), "temporal weight must be less or equal to 1!"

        # Define some hyperparameters
        self.w_temp = w_temp
        self.t_clust = t_clust
        self.verbose = verbose

        # Load data prepared for training
        self.df_train, self.df_test = prepare_data()
        self.states = self.df_train['state'].unique()

        # Define target variables
        self.target_lst = ["positiveIncrease", "hospitalizedIncrease", "deathIncrease"]
        # Select additional regressors with structural data information
        self.est_regressors = self.df_train.drop(columns=(['ds', 'state'] + self.target_lst)).columns.tolist()

        self.model_dir = Path(__file__).parent.parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def cluster_states(self):
        """
        Returns a dataframe with the allocation of states to clusters with similarity
        """

        def load_data():
            """
            Returns the states' distances datasets, wrt. both temporal and estructural data
            """
            # Resolve based on this script’s location (not the CWD)
            base_path = Path(__file__).parent.parent / "data/preprocessed/dataMatrix/state_distances"

            df_tmp_DM = {
                'positiveIncrease': pd.read_csv(base_path / "tmp_dist_Pos.csv"), 
                'hospitalizedIncrease': pd.read_csv(base_path / "tmp_dist_Hosp.csv"), 
                'deathIncrease': pd.read_csv(base_path / "tmp_dist_Death.csv")
            } # one component for each target variable
            df_est_DM = pd.read_csv(base_path / "est_dist.csv")
            
            return df_tmp_DM, df_est_DM
        
        # Load necessary data for the clustering, which is the computed distances
        # DTW for temporal data and euclidean for estructural
        df_tmp_DM, df_est_DM = load_data()

        def _toMatrix(df):
            """
            Converts from df to distance matrix
            """
            return df.iloc[:, 1:].values

        def agg_clustering(temp_comp_DM):
            """
            Aggregates states according to a customed distance matrix, that accounts for both 
            temporal and estructual data

            Args:
                temp_comp_DM: distance matrix of a temporal component to perform the aggregtion with structural DM
                temp_comp_name: str representing the temporal component
            """

            # Normalize each matrix to [0, 1]
            scaler = MinMaxScaler()
            d_temp_normalized = scaler.fit_transform(_toMatrix(temp_comp_DM)) # temporal component DM
            d_est_normalized = scaler.fit_transform(_toMatrix(df_est_DM)) # structural DM

            # d(A, B) = w_temp * d_temp(A, B) + w_est * d_est(A, B)
            dist_matrix_agg = self.w_temp * d_temp_normalized + (1 - self.w_temp) * d_est_normalized

            # Compute linkage matrix (using Ward's method)
            Z = linkage(dist_matrix_agg, method='ward')

            # Cut the dendrogram at distance threshold t_clust
            clusters = fcluster(Z, t=self.t_clust, criterion='distance')

            # Calculate Cophenetic Correlation Coefficient -> original pairwise distances preservation
            coph_dist = cophenet(Z)
            original_dist = pdist(dist_matrix_agg)

            correlation = np.corrcoef(original_dist, coph_dist)[0, 1]
            print(f"Cophenetic Correlation: {correlation:.3f}")

            return Z, clusters
            
        def show_clustering(Z, c_name):
            """
            Function to print or store the resulting dendogram from the clustering

            Args:
                Z: hierarchical clustering encoded with a linkage matrix
                c_name: str w. target variable object of the clustering
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
                base_path = Path(__file__).parent.parent / "src_models/experiments_prophet"
                plt.savefig(f"{base_path / f'clust_{c_name}.png'}", dpi=300, bbox_inches='tight')
                
        # Perform clustering for each component
        cluster_comp = dict()
        for c_name, c_DM in df_tmp_DM.items():

            # Compute clustering
            Z, clusters = agg_clustering(c_DM)

            # Visualize results
            show_clustering(Z, c_name)

            # Store results
            cluster_comp[c_name] = pd.DataFrame({
                "state": self.states,
                c_name: clusters
            })
        
        # Show number of clusters per target variable
        for c_name, clust_df in cluster_comp.items():
            print(f"Nº of clusters for {c_name}: {len(clust_df[c_name].unique())}")

        # Unify results in a single df
        self.cluster_df = cluster_comp['positiveIncrease'].merge(cluster_comp['hospitalizedIncrease'], on='state') \
                .merge(cluster_comp['deathIncrease'], on='state')

        # Export dataframe
        base_path = Path(__file__).parent.parent / "src_models/experiments_prophet"
        self.cluster_df.to_csv(f"{base_path / 'state_clusters.csv'}", index=False)


    def train_prophet(self):
        """
        Trains 3 prophet models for each of the clusters, one per target variable
        """
    
        # Get the state-cluster dataframe so as to group states timeseries
        cluster_df = self.cluster_df

        self.models = dict()
        # Iterate through targets and clusters
        for tget in self.target_lst:
            clusters = cluster_df[tget].unique()
            for clust in clusters:
                model_key = f"{tget}_cluster{clust}"
                model_path = self.model_dir / f"{model_key}.json"
                # Check if the model exists
                if model_path.exists():
                    print(f"Loading existing model: {model_key}")
                    with open(model_path, 'r') as fin:
                        model = model_from_json(fin.read())
                    self.models[model_key] = model
                    continue
                # Train if it doesn't exist
                # States that belong to clust for tget
                sts_clust = cluster_df[cluster_df[tget] == clust]['state'].unique()

                # Select columns for training
                columns = ['ds', tget] + self.est_regressors

                # Get the desired dataframe: from cluster states' data and selected cols
                df_clust = self.df_train[self.df_train['state'].isin(sts_clust)]
                df_prophet = df_clust[columns].rename(columns={tget: 'y'})

                model = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0
                )

                # Add regressors
                for reg in self.est_regressors:
                    model.add_regressor(reg)
                
                # Train the Prophet model
                model.fit(df_prophet)
                print(f"Modelo entrenado para {model_key}")
                # Guardar modelo
                with open(model_path, 'w') as fout:
                    fout.write(model_to_json(model))
                
                self.models[model_key] = model

    def eval_prophet(self):
        """
        Evaluate performance using the test set (comparison with real data)
        """
        metrics = []
        
        for state in self.states:
            state_test = self.df_test[self.df_test['state'] == state]
            
            if state_test.empty:
                continue
            
            # Test dates (first 7 days)
            test_dates = state_test['ds'].unique()[:7]
            df_test_7d = state_test[state_test['ds'].isin(test_dates)]
            
            for target in self.target_lst:
                cluster = self.cluster_df.loc[self.cluster_df['state'] == state, target].values[0]
                model_key = f"{target}_cluster{cluster}"
                
                if model_key not in self.models:
                    continue
                
                # Predict
                future = self._prepare_future(state, test_dates[0])
                forecast = self.models[model_key].predict(future)
                pred_sum = forecast['yhat'].sum()
                
                # Calculate metrics
                y_true = df_test_7d[target].sum()
                mae = mean_absolute_error([y_true], [pred_sum])
                rmse = np.sqrt(mae)
                mape = (abs(y_true - pred_sum)/y_true*100) if y_true != 0 else 0.0

                metrics.append({
                    'target': target,
                    'state': state,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
        
        # Show results
        print("\nPerformance evaluation:")
        for m in metrics:
            print(
                f"{m['target']} & {m['state']}, "
                f"MAE: {m['MAE']:.2f}, "
                f"RMSE: {m['RMSE']:.2f}, "
                f"MAPE: {m['MAPE']:.2f}%"
            )

    def generate_prediction_matrix(self):
        """
        Generates the matrix of future predictions (7 days after the last training data)
        """
        prediction_matrix = []
        
        for state in self.states:
            state_data = {'state': state}
            
            # Prepare for the future (7 days post-training)
            last_train_date = pd.to_datetime(
                self.df_train[self.df_train['state'] == state]['ds'].max()
            )
            future_dates = pd.date_range(
                start=last_train_date + timedelta(days=1),
                periods=7,
                freq='D'
            )
            
            future = self._prepare_future(state, future_dates[0])
            
            for target in self.target_lst:
                cluster = self.cluster_df.loc[self.cluster_df['state'] == state, target].values[0]
                model_key = f"{target}_cluster{cluster}"
                
                if model_key not in self.models:
                    state_data[target] = None
                    continue
                
                forecast = self.models[model_key].predict(future)
                state_data[target] = forecast['yhat'].sum()
            
            prediction_matrix.append(state_data)
        
        # Save and Return
        output_path = self.model_dir.parent / "output_modelos/predictions_matrix.csv"
        pd.DataFrame(prediction_matrix).to_csv(output_path, index=False)
        print(f"\nPrediction matrix saved in: {output_path}")
        
        return prediction_matrix

    def _prepare_future(self, state: str, start_date: pd.Timestamp) -> pd.DataFrame:
        """
        Helper to prepare future data (reusable in both methods)
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
    trainer = ProphetTrainer()
    trainer.cluster_states()
    trainer.train_prophet()
    trainer.eval_prophet()
    trainer.generate_prediction_matrix()
