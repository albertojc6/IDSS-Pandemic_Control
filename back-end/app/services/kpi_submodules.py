import pandas as pd
import plotly.express as px
from typing import Optional, Tuple
from datetime import datetime
from app.models import PandemicData  # Add this import at the top
import numpy as np


class RecommendationTracker:
    def __init__(self):
        # Initialize information dataframe
        self.historic = pd.DataFrame({
            'date': pd.Series(dtype='datetime64[ns, UTC]'),
            'rec_id': pd.Series(dtype='int'),
            'state': pd.Series(dtype='str'),
            'taken': pd.Series(dtype='boolean')
        })

    def account_recommendation(self, rec_info: Tuple[str, int, str]):
        """
        Once a recommendation is made, this function is called to store its information.

        params:
        rec_info: Tuple containing:
            - date: Date of the recommendation (string or datetime)
            - rec_id: ID of the recommendation
            - state: State that received the recommendation
        """
        formatted_info = {
            'date': pd.to_datetime(rec_info[0], utc=True),
            'rec_id': rec_info[1],
            'state': rec_info[2],
            'taken': False  # Default to False, to be updated by mark_as_taken
        }

        self.historic = pd.concat([self.historic, pd.DataFrame([formatted_info])], ignore_index=True)

    def mark_as_taken(self, rec_id: int):
        """
        Mark a recommendation as taken.
        """
        self.historic.loc[self.historic['rec_id'] == rec_id, 'taken'] = True

    def plot_weekly_taken_ratio(self,
                                state: Optional[str] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None):
        """
        Returns a Plotly plot of the weekly ratio of recommendations taken.

        :param state: Optional state to filter by
        :param start_date: Optional start date (inclusive) in 'YYYY-MM-DD' format
        :param end_date: Optional end date (inclusive) in 'YYYY-MM-DD' format
        :return: Plotly figure object
        """
        df = self.historic.copy()

        # Apply filters
        if state:
            df = df[df['state'] == state]

        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date, utc=True)]

        if df.empty:
            raise ValueError("No data available for the selected filters.")

        df['week'] = df['date'].dt.to_period('W').apply(lambda r: pd.Timestamp(r.start_time, tz='UTC'))

        weekly_stats = df.groupby('week').agg(
            total_recommendations=('taken', 'count'),
            recommendations_taken=('taken', 'sum')
        )
        # Ensure 'recommendations_taken' is float for division if it's a boolean sum
        weekly_stats['ratio'] = weekly_stats['recommendations_taken'].astype(float) / weekly_stats[
            'total_recommendations']

        # Plot
        fig = px.line(weekly_stats, x=weekly_stats.index, y='ratio',
                      labels={'x': 'Week', 'ratio': 'Taken Ratio'},
                      title='Weekly Recommendation Taken Ratio')
        fig.update_layout(xaxis_title='Week', yaxis_title='Taken Ratio', yaxis_range=[0, 1])
        return fig


class PrecisionTracker:
    """
    Tracks predicted vs actual hospital saturation on a state and daily basis,
    Analysis of model precision (i.e., prediction - actual).
    """

    def __init__(self):
        # Initialize empty DataFrame
        self.data = pd.DataFrame(columns=[
            'date', 'state', 'type', 'value'
        ])
        self.data['date'] = pd.to_datetime(self.data['date'], utc=True)  # Ensure UTC type

    def register_prediction(self, date: str, state: str, predicted_value: float):
        """
        Registers a predicted hospital saturation value for a given state and date.

        :param date: The prediction date (format: 'YYYY-MM-DD')
        :param state: The state name
        :param predicted_value: The predicted saturation value
        """
        self._register_entry(date, state, 'prediction', predicted_value)

    def register_actual(self, date: str, state: str, actual_value: float):
        """
        Registers an actual hospital saturation value for a given state and date.

        :param date: The actual observation date (format: 'YYYY-MM-DD')
        :param state: The state name
        :param actual_value: The actual saturation value
        """
        self._register_entry(date, state, 'actual', actual_value)

    def _register_entry(self, date: str, state: str, entry_type: str, value: float):
        entry = {
            'date': pd.to_datetime(date, utc=True),  # Ensure UTC
            'state': state,
            'type': entry_type,
            'value': value
        }
        self.data = pd.concat([self.data, pd.DataFrame([entry])], ignore_index=True)

    def plot_cases_precision(self, state: Optional[str] = None):
        """
        Plots the difference between predicted and actual saturation per day
        (i.e., prediction error). Optionally filters by state.

        :param state: Optional state to filter the data
        :return: A Plotly line chart figure
        """
        df = self.data.copy()

        if state:
            df = df[df['state'] == state]

        if df.empty:
            raise ValueError("No data to display for the selected state or filters.")

        pivot_df = df.pivot_table(
            index=['date', 'state'],
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Only keep rows with both prediction and actual values present
        pivot_df.dropna(subset=['prediction', 'actual'], inplace=True)

        if pivot_df.empty:
            raise ValueError("No data with both prediction and actual values for the selected state or filters.")

        pivot_df['error'] = pivot_df['prediction'] - pivot_df['actual']

        # Plot
        fig = px.line(pivot_df, x='date', y='error', color='state',
                      title='Prediction Error (Prediction - Actual)',
                      labels={'error': 'Error (Prediction - Actual)', 'date': 'Date'})
        fig.update_layout(xaxis_title='Date', yaxis_title='Error (Prediction - Actual)')
        return fig


class PreventedTracker:
    """
    Tracks start of week 7 day ahead prediction vs actual values of hospital saturation,
    Analysis of prevented hospital saturation with system reccomendations (i.e., prediction - actual).
    """

    def __init__(self):
        self.data = pd.DataFrame(columns=[
            'pred_date', 'target_date', 'state', 'type', 'value'
        ])
        # Ensure date columns are of datetime type with UTC, even if DataFrame is initially empty
        self.data['pred_date'] = pd.to_datetime(self.data['pred_date'], utc=True)
        self.data['target_date'] = pd.to_datetime(self.data['target_date'], utc=True)

    def register_prediction(self, pred_date: str, target_date: str, state: str, predicted_value: float):
        """
        Registers a predicted hospital saturation value for a given state and date.

        :param pred_date: The date the prediction was made (format: 'YYYY-MM-DD')
        :param target_date: The future date the prediction is for (format: 'YYYY-MM-DD')
        :param state: The state name
        :param predicted_value: The predicted saturation value
        """
        self._register_entry(pred_date, target_date, state, 'prediction', predicted_value)

    def register_actual(self, pred_date: str, target_date: str, state: str, actual_value: float):
        """
        Registers an actual hospital saturation value for a given state and date,
        corresponding to a prior prediction.

        :param pred_date: The date the original prediction was made (format: 'YYYY-MM-DD')
        :param target_date: The date of actual observation (format: 'YYYY-MM-DD')
        :param state: The state name
        :param actual_value: The actual saturation value
        """
        self._register_entry(pred_date, target_date, state, 'actual', actual_value)

    def _register_entry(self, pred_date: str, target_date: str, state: str, entry_type: str, value: float):
        entry = {
            'pred_date': pd.to_datetime(pred_date, utc=True),
            'target_date': pd.to_datetime(target_date, utc=True),
            'state': state,
            'type': entry_type,
            'value': value
        }
        self.data = pd.concat([self.data, pd.DataFrame([entry])], ignore_index=True)

    def plot_prevented_saturation(self,
                                  state: Optional[str] = None,
                                  start_target_date: Optional[str] = None,
                                  end_target_date: Optional[str] = None):
        """
        Plots the difference between 7-day ahead predicted and actual saturation
        for the target date (i.e., prevented saturation).
        Optionally filters by state and target date range.

        :param state: Optional state to filter the data.
        :param start_target_date: Optional start target date (inclusive) in 'YYYY-MM-DD' format.
        :param end_target_date: Optional end target date (inclusive) in 'YYYY-MM-DD' format.
        :return: A Plotly line chart figure.
        """
        df_plot = self.data.copy()

        # Apply filters
        if state:
            df_plot = df_plot[df_plot['state'] == state]
        if start_target_date:
            df_plot = df_plot[df_plot['target_date'] >= pd.to_datetime(start_target_date, utc=True)]
        if end_target_date:
            df_plot = df_plot[df_plot['target_date'] <= pd.to_datetime(end_target_date, utc=True)]

        if df_plot.empty:
            raise ValueError("No data available for the selected filters.")

        pivot_df = df_plot.pivot_table(
            index=['target_date', 'state', 'pred_date'],
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        pivot_df.dropna(subset=['prediction', 'actual'], inplace=True)

        if pivot_df.empty:
            raise ValueError("No data with both prediction and actual values for the selected filters.")

        pivot_df['prevented_saturation'] = pivot_df['prediction'] - pivot_df['actual']

        plot_data = pivot_df.sort_values(by=['state', 'target_date'])

        fig_title = 'Prevented Hospital Saturation (Prediction - Actual)'
        if state:
            fig_title += f" for {state}"

        fig = px.line(plot_data, x='target_date', y='prevented_saturation', color='state',
                      title=fig_title,
                      labels={'prevented_saturation': 'Prevented Saturation', 'target_date': 'Target Date'},
                      markers=True)
        fig.update_layout(xaxis_title='Target Date', yaxis_title='Prevented Saturation')
        return fig


class SatisfactionTracker:
    """
    Tracks satisfaction scores (1-5 stars) for each state.
    Analysis of actual system overall satisfaction.
    """

    def __init__(self, state=None, start_date=None, end_date=None):
        self.data = pd.DataFrame({
            'date': pd.Series(dtype='datetime64[ns, UTC]'),
            'state': pd.Series(dtype='str'),
            'score': pd.Series(dtype='int')  # 1-5 stars
        })
        self._initialize_default_ratings(state, start_date, end_date)

    def _initialize_default_ratings(self, state, start_date, end_date):
        # Fetch all dates from PandemicData for this state and range
        query = PandemicData.query
        if state:
            query = query.filter(PandemicData.state == state)
        if start_date:
            query = query.filter(PandemicData.date >= start_date)
        if end_date:
            query = query.filter(PandemicData.date <= end_date)
        dates = [row.date for row in query.order_by(PandemicData.date).all()]
        default_ratings = []
        for i, date in enumerate(dates):
            score = 5 if i % 2 == 0 else 4
            default_ratings.append({
                'date': pd.Timestamp(date, tz='UTC'),
                'state': state or 'Default',
                'score': score
            })
        if default_ratings:
            default_df = pd.DataFrame(default_ratings)
            self.data = pd.concat([self.data, default_df], ignore_index=True)
            self.data = self.data.sort_values('date', ascending=False)

    def register_satisfaction_score(self, date: str, state: str, score: int):
        """
        Registers a satisfaction score for a given state and date.

        :param date: The date of the satisfaction rating (format: 'YYYY-MM-DD').
        :param state: The state name.
        :param score: The satisfaction score (integer, 1-5 stars).
        :raises ValueError: If the score is not an integer between 1 and 5.
        """
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError("Satisfaction score must be an integer between 1 and 5 stars.")

        # Convert date to UTC datetime
        date_dt = pd.to_datetime(date, utc=True)
        
        # Remove any existing score for this date and state
        self.data = self.data[
            ~((self.data['date'].dt.date == date_dt.date()) & 
              (self.data['state'] == state))
        ]

        # Add the new score
        new_entry_df = pd.DataFrame([{
            'date': date_dt,
            'state': state,
            'score': score
        }])

        self.data = pd.concat([self.data, new_entry_df], ignore_index=True)
        self.data = self.data.sort_values('date', ascending=False)

    def plot_satisfaction_trends(self,
                               state: Optional[str] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               aggregation_period: str = 'D'):
        """
        Plots the trend of satisfaction scores over time.

        :param state: Optional state to filter by. If None, shows trends for all states.
        :param start_date: Optional start date (inclusive) in 'YYYY-MM-DD' format.
        :param end_date: Optional end date (inclusive) in 'YYYY-MM-DD' format.
        :param aggregation_period: Pandas offset alias for aggregation ('D' for daily).
        :return: Chart.js compatible data object.
        """
        df_plot = self.data.copy()

        # If we have default ratings and a state is specified, replace 'Default' with the actual state
        if state and 'Default' in df_plot['state'].values:
            df_plot.loc[df_plot['state'] == 'Default', 'state'] = state

        # Apply filters
        if state:
            df_plot = df_plot[df_plot['state'] == state]
        if start_date:
            df_plot = df_plot[df_plot['date'] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df_plot = df_plot[df_plot['date'] <= pd.to_datetime(end_date, utc=True)]

        if df_plot.empty:
            return {
                'labels': [],
                'datasets': [{
                    'label': 'Satisfaction Rating',
                    'data': [],
                    'borderColor': 'rgba(255, 206, 86, 1)',
                    'backgroundColor': 'rgba(255, 206, 86, 0.2)',
                    'fill': True,
                    'tension': 0.1
                }]
            }

        # Sort by date to ensure chronological order
        df_plot = df_plot.sort_values('date')

        return {
            'labels': df_plot['date'].dt.strftime('%Y-%m-%d').tolist(),
            'datasets': [{
                'label': 'Satisfaction Rating',
                'data': df_plot['score'].tolist(),
                'borderColor': 'rgba(255, 206, 86, 1)',
                'backgroundColor': 'rgba(255, 206, 86, 0.2)',
                'fill': True,
                'tension': 0.1
            }]
        }


class PredictionKPITracker:
    """
    Tracks prediction precision and prevented saturation KPIs.
    For MVP purposes, this generates mock error distributions based on specified probabilities.
    """
    
    def __init__(self):
        self.precision_distribution = {
            '10_percent': 0.70,  # 70% of cases have ±10% error
            '20_percent': 0.25,  # 25% of cases have ±20% error
            '50_percent': 0.05   # 5% of cases have ±50% error
        }
        
        self.prevented_distribution = {
            '50_percent': 0.70,  # 70% of cases have 50% prevented
            '30_percent': 0.10,  # 10% of cases have 30% prevented
            '15_percent': 0.10,  # 10% of cases have 15% prevented
            '10_percent': 0.05,  # 5% of cases have 10% prevented
            '0_percent': 0.05    # 5% of cases have 0% prevented
        }
    
    def get_prediction_error(self, prediction_value: float) -> float:
        """
        Generate a mock prediction error based on the specified distribution.
        
        Args:
            prediction_value: The predicted value
            
        Returns:
            float: The error percentage (positive or negative)
        """
        # Generate random number to determine which error range to use
        rand = np.random.random()
        cumulative = 0
        
        # Determine error range based on probabilities
        if rand < self.precision_distribution['10_percent']:
            error_range = 0.10
        elif rand < self.precision_distribution['10_percent'] + self.precision_distribution['20_percent']:
            error_range = 0.20
        else:
            error_range = 0.50
        
        # Generate random error within the range
        error = np.random.uniform(-error_range, error_range)
        
        # Calculate actual error value
        return prediction_value * error
    
    def get_prevented_saturation(self, prediction_value: float) -> float:
        """
        Generate a mock prevented saturation percentage based on the specified distribution.
        
        Args:
            prediction_value: The predicted value
            
        Returns:
            float: The prevented saturation percentage (0-50%)
        """
        # Generate random number to determine which prevention percentage to use
        rand = np.random.random()
        cumulative = 0
        
        # Determine prevention percentage based on probabilities
        if rand < self.prevented_distribution['50_percent']:
            prevention = 0.50
        elif rand < self.prevented_distribution['50_percent'] + self.prevented_distribution['30_percent']:
            prevention = 0.30
        elif rand < (self.prevented_distribution['50_percent'] + 
                    self.prevented_distribution['30_percent'] + 
                    self.prevented_distribution['15_percent']):
            prevention = 0.15
        elif rand < (self.prevented_distribution['50_percent'] + 
                    self.prevented_distribution['30_percent'] + 
                    self.prevented_distribution['15_percent'] + 
                    self.prevented_distribution['10_percent']):
            prevention = 0.10
        else:
            prevention = 0.00
        
        # Calculate actual prevented value
        return prediction_value * prevention