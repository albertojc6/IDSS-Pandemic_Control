import pandas as pd
import plotly.express as px
from typing import Optional, Tuple
from datetime import datetime


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
    Tracks satisfaction imputed daily by each state 0-5 stars.
    Analysis of actual system overall satisfaction.
    """

    def __init__(self):
        self.data = pd.DataFrame({
            'date': pd.Series(dtype='datetime64[ns, UTC]'),
            'state': pd.Series(dtype='str'),
            'score': pd.Series(dtype='int') #0-5
        })

    def register_satisfaction_score(self, date: str, state: str, score: int):
        """
        Registers a satisfaction score for a given state and date.

        :param date: The date of the satisfaction rating (format: 'YYYY-MM-DD').
        :param state: The state name.
        :param score: The satisfaction score (integer, 0-5).
        :raises ValueError: If the score is not an integer between 0 and 5.
        """
        if not isinstance(score, int) or not (0 <= score <= 5):
            raise ValueError("Satisfaction score must be an integer between 0 and 5.")

        new_entry_df = pd.DataFrame([{
            'date': pd.to_datetime(date, utc=True),
            'state': state,
            'score': score
        }])

        for col, dtype in {'date': 'datetime64[ns, UTC]', 'state': 'str', 'score': 'int'}.items():
            if col in new_entry_df:
                new_entry_df[col] = new_entry_df[col].astype(dtype)

        self.data = pd.concat([self.data, new_entry_df], ignore_index=True)

    def plot_satisfaction_trends(self,
                                 state: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 aggregation_period: str = 'W'):
        """
        Plots the trend of satisfaction scores over time.

        :param state: Optional state to filter by. If None, shows trends for all states.
        :param start_date: Optional start date (inclusive) in 'YYYY-MM-DD' format.
        :param end_date: Optional end date (inclusive) in 'YYYY-MM-DD' format.
        :param aggregation_period: Pandas offset alias for aggregation ('D' for daily,
                                   'W' for weekly, 'M' for monthly). Default 'W'.
        :return: Plotly figure object.
        :raises ValueError: If no data is available for the selected filters.
        """
        df_plot = self.data.copy()

        # Apply filters
        if state:
            df_plot = df_plot[df_plot['state'] == state]
        if start_date:
            df_plot = df_plot[df_plot['date'] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df_plot = df_plot[df_plot['date'] <= pd.to_datetime(end_date, utc=True)]

        if df_plot.empty:
            raise ValueError("No satisfaction data available for the selected filters.")

        # Create the time period column for aggregation
        df_plot['period_start'] = df_plot['date'].dt.to_period(aggregation_period).apply(
            lambda r: pd.Timestamp(r.start_time, tz='UTC'))

        # Group by this new period column and 'state', calculate mean score
        agg_stats = df_plot.groupby(['period_start', 'state'])['score'].mean().reset_index()

        # Determine plot titles and labels based on aggregation period
        period_map = {'D': "Daily", 'W': "Weekly", 'M': "Monthly"}
        title_prefix = period_map.get(aggregation_period.upper(), f"Average (Period: {aggregation_period})")
        x_axis_label = period_map.get(aggregation_period.upper(), f"Period Start ({aggregation_period})")

        fig_title = f"{title_prefix} Average Satisfaction Score"
        if state:  # If a specific state was filtered, reflect it in the title
            fig_title += f" for {state}"
        elif len(agg_stats['state'].unique()) == 1:  # If only one state's data remains after other filters
            single_state_name = agg_stats['state'].unique()[0]
            fig_title += f" for {single_state_name}"

        fig = px.line(agg_stats, x='period_start', y='score', color='state',
                      title=fig_title,
                      labels={'period_start': x_axis_label, 'score': 'Average Satisfaction (0-5)'},
                      markers=True)
        fig.update_layout(xaxis_title=x_axis_label, yaxis_title='Average Satisfaction (0-5)',
                          yaxis_range=[0, 5.1])
        return fig