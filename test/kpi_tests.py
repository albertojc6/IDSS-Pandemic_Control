import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.express as px

from kpi.kpi_submodules import RecommendationTracker, PrecisionTracker, PreventedTracker, SatisfactionTracker


######################################################################################
#######                       RECOMMENDATIONS TAKEN                             ######
######################################################################################

@pytest.fixture
def rec_tracker():
    tracker = RecommendationTracker()
    # Ensure UTC
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(10):
        date = base_date + timedelta(days=i)
        state = 'CA' if i < 5 else 'NY'
        tracker.account_recommendation((date.isoformat(), i, state))
        if i % 2 == 0:  # rec_ids 0, 2, 4, 6, 8 are taken
            tracker.mark_as_taken(i)
    return tracker


def test_recommendation_tracker_initialization():
    tracker = RecommendationTracker()
    assert tracker.historic.empty
    assert list(tracker.historic.columns) == ['date', 'rec_id', 'state', 'taken']
    assert tracker.historic['date'].dtype == 'datetime64[ns, UTC]'
    assert tracker.historic['taken'].dtype == 'boolean'


def test_account_recommendation(rec_tracker):
    assert len(rec_tracker.historic) == 10
    assert rec_tracker.historic.iloc[0]['state'] == 'CA'
    assert rec_tracker.historic.iloc[5]['state'] == 'NY'
    assert pd.Timestamp(rec_tracker.historic.iloc[0]['date']) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert rec_tracker.historic.iloc[0]['taken'] == True  # rec_id 0
    assert rec_tracker.historic.iloc[1]['taken'] == False  # rec_id 1


def test_mark_as_taken(rec_tracker):
    # rec_id 1 was initially False
    assert rec_tracker.historic[rec_tracker.historic['rec_id'] == 1]['taken'].iloc[0] == False
    rec_tracker.mark_as_taken(1)
    assert rec_tracker.historic[rec_tracker.historic['rec_id'] == 1]['taken'].iloc[0] == True


def test_weekly_taken_ratio_all_data(rec_tracker):
    fig = rec_tracker.plot_weekly_taken_ratio()
    assert fig is not None
    data = fig.data[0]

    # Week 1: 2024-01-01 to 2024-01-07 (covers rec_ids 0-6)
    # CA: recs 0,1,2,3,4. Taken: 0,2,4 (3 taken)
    # NY: recs 5,6. Taken: 6 (1 taken)
    # Total in first week (days 0-6, 7 recs): 0,1,2,3,4,5,6. Taken: 0,2,4,6 (4 taken). Ratio = 4/7
    # Week 2: 2024-01-08 to 2024-01-14 (covers rec_ids 7-9)
    # NY: recs 7,8,9. Taken: 8 (1 taken)
    # Total in second week (days 7-9, 3 recs): 7,8,9. Taken: 8 (1 taken). Ratio = 1/3

    assert len(data.x) == 2  # Two weeks of data
    expected_ratio_w1 = (3 + 1) / 7.0  # 4 taken / 7 total in week 1
    expected_ratio_w2 = 1 / 3.0  # 1 taken / 3 total in week 2
    assert pytest.approx(data.y[0], 0.01) == expected_ratio_w1
    assert pytest.approx(data.y[1], 0.01) == expected_ratio_w2
    assert fig.layout.yaxis.title.text == 'Taken Ratio'


def test_weekly_taken_ratio_filtered_by_state(rec_tracker):
    # State CA: recs 0,1,2,3,4 (all in first week). Taken: 0,2,4 (3 taken). Ratio 3/5 = 0.6
    fig_ca = rec_tracker.plot_weekly_taken_ratio(state='CA')
    data_ca = fig_ca.data[0]
    assert len(data_ca.x) == 1
    assert pytest.approx(data_ca.y[0], 0.01) == 0.6

    # State NY: recs 5,6 (first week), 7,8,9 (second week)
    # Week 1 (NY): recs 5,6. Taken: 6 (1 taken). Ratio 1/2 = 0.5
    # Week 2 (NY): recs 7,8,9. Taken: 8 (1 taken). Ratio 1/3
    fig_ny = rec_tracker.plot_weekly_taken_ratio(state='NY')
    data_ny = fig_ny.data[0]
    assert len(data_ny.x) == 2
    assert pytest.approx(data_ny.y[0], 0.01) == 0.5
    assert pytest.approx(data_ny.y[1], 0.01) == 1 / 3


def test_weekly_taken_ratio_filtered_by_date(rec_tracker):
    # Dates: 2024-01-04 (day 3, CA, not taken), 2024-01-05 (day 4, CA, taken),
    # 2024-01-06 (day 5, NY, not taken), 2024-01-07 (day 6, NY, taken).
    # All fall into the same week (starting 2024-01-01).
    # Recs in this range: 4 (IDs 3,4,5,6). Taken in this range: 2 (IDs 4,6). Ratio = 2/4 = 0.5.
    fig = rec_tracker.plot_weekly_taken_ratio(start_date='2024-01-04', end_date='2024-01-07')
    data = fig.data[0]
    assert len(data.x) == 1
    assert pytest.approx(data.y[0], 0.01) == 0.5


def test_recommendation_tracker_empty_result_raises_error(rec_tracker):
    with pytest.raises(ValueError, match="No data available for the selected filters."):
        rec_tracker.plot_weekly_taken_ratio(start_date='2030-01-01', end_date='2030-01-10')
    with pytest.raises(ValueError, match="No data available for the selected filters."):
        rec_tracker.plot_weekly_taken_ratio(state='TX')


######################################################################################
#######                       PRECISION TRACKER (Model Error)                   ######
######################################################################################

@pytest.fixture
def precision_tracker_fixture():
    pt = PrecisionTracker()
    # All dates are implicitly UTC due to pd.to_datetime(date, utc=True) in _register_entry
    pt.register_prediction("2024-01-01", "CA", 150)
    pt.register_actual("2024-01-01", "CA", 100)  # CA Error: 150-100 = 50
    pt.register_prediction("2024-01-02", "CA", 120)
    pt.register_actual("2024-01-02", "CA", 130)  # CA Error: 120-130 = -10

    pt.register_prediction("2024-01-01", "NY", 80)  # NY only has prediction
    pt.register_actual("2024-01-01", "NY", 70)  # NY Error: 80-70 = 10

    pt.register_prediction("2024-01-03", "TX", 200)  # TX only has prediction
    return pt


def test_precision_tracker_register_entries(precision_tracker_fixture):
    df = precision_tracker_fixture.data
    assert len(df) == 7
    assert set(df['type'].unique()) == {'prediction', 'actual'}
    assert 'value' in df.columns
    assert df['date'].dtype == 'datetime64[ns, UTC]'
    assert df[df['state'] == 'CA'].shape[0] == 4
    assert df[df['state'] == 'NY'].shape[0] == 2
    assert df[df['state'] == 'TX'].shape[0] == 1


def test_plot_cases_precision_all_states(precision_tracker_fixture):
    fig = precision_tracker_fixture.plot_cases_precision()
    assert fig is not None
    # Two states should have lines: CA and NY (TX does not have actual)
    assert len(fig.data) == 2  # CA and NY

    ca_data = next(trace for trace in fig.data if trace.name == 'CA')
    ny_data = next(trace for trace in fig.data if trace.name == 'NY')

    assert list(ca_data.y) == [50, -10]

    assert list(ny_data.y) == [10]
    assert fig.layout.yaxis.title.text == 'Error (Prediction - Actual)'


def test_plot_cases_precision_filtered_by_state(precision_tracker_fixture):
    fig_ca = precision_tracker_fixture.plot_cases_precision(state="CA")
    assert len(fig_ca.data) == 1
    assert fig_ca.data[0].name == "CA"
    assert list(fig_ca.data[0].y) == [50, -10]

    # For NY, it has both prediction and actual, so it should plot
    fig_ny = precision_tracker_fixture.plot_cases_precision(state="NY")
    assert len(fig_ny.data) == 1
    assert fig_ny.data[0].name == "NY"
    assert list(fig_ny.data[0].y) == [10]


def test_plot_cases_precision_raises_on_no_paired_data(precision_tracker_fixture):
    # TX only has a prediction, no actual. So no pair to calculate error.
    with pytest.raises(KeyError):
        precision_tracker_fixture.plot_cases_precision(state="TX")


def test_plot_cases_precision_raises_on_unknown_state():
    pt = PrecisionTracker()
    pt.register_prediction("2024-01-01", "CA", 150)
    pt.register_actual("2024-01-01", "CA", 100)
    with pytest.raises(ValueError, match="No data to display for the selected state or filters."):  # This error first
        pt.plot_cases_precision(state="FL")  # FL has no data at all


######################################################################################
#######                       PREVENTED TRACKER                                 ######
######################################################################################

@pytest.fixture
def prevented_tracker_fixture():
    pt = PreventedTracker()
    # pred_date, target_date, state, value
    # Scenario 1: CA, prediction made on Jan 1 for Jan 8
    pt.register_prediction("2024-01-01", "2024-01-08", "CA", 200)
    pt.register_actual("2024-01-01", "2024-01-08", "CA", 150)  # Prevented: 50

    # Scenario 2: CA, prediction made on Jan 8 for Jan 15
    pt.register_prediction("2024-01-08", "2024-01-15", "CA", 220)
    pt.register_actual("2024-01-08", "2024-01-15", "CA", 230)  # Prevented: -10

    # Scenario 3: NY, prediction made on Jan 1 for Jan 8
    pt.register_prediction("2024-01-01", "2024-01-08", "NY", 100)
    pt.register_actual("2024-01-01", "2024-01-08", "NY", 80)  # Prevented: 20

    # Scenario 4: TX, only prediction
    pt.register_prediction("2024-01-01", "2024-01-08", "TX", 120)

    # Scenario 5: FL, prediction for different target date
    pt.register_prediction("2024-01-02", "2024-01-09", "FL", 90)
    pt.register_actual("2024-01-02", "2024-01-09", "FL", 90)  # Prevented: 0
    return pt


def test_prevented_tracker_register_entries(prevented_tracker_fixture):
    df = prevented_tracker_fixture.data
    assert len(df) == 9
    assert set(df['type'].unique()) == {'prediction', 'actual'}
    assert 'value' in df.columns
    assert df['pred_date'].dtype == 'datetime64[ns, UTC]'
    assert df['target_date'].dtype == 'datetime64[ns, UTC]'
    assert df[df['state'] == 'CA'].shape[0] == 4
    assert df[df['state'] == 'NY'].shape[0] == 2
    assert df[df['state'] == 'TX'].shape[0] == 1
    assert df[df['state'] == 'FL'].shape[0] == 2


def test_plot_prevented_saturation_all_states(prevented_tracker_fixture):
    fig = prevented_tracker_fixture.plot_prevented_saturation()
    assert fig is not None
    assert len(fig.data) == 3  # CA, NY, FL have pairs. TX does not.

    assert fig.layout.yaxis.title.text == 'Prevented Saturation'
    assert "Prevented Hospital Saturation" in fig.layout.title.text


def test_plot_prevented_saturation_filtered_by_state(prevented_tracker_fixture):
    fig_ca = prevented_tracker_fixture.plot_prevented_saturation(state="CA")
    assert len(fig_ca.data) == 1
    assert fig_ca.data[0].name == "CA"
    assert list(fig_ca.data[0].y) == [50, -10]
    assert "for CA" in fig_ca.layout.title.text


def test_plot_prevented_saturation_filtered_by_target_date(prevented_tracker_fixture):
    fig = prevented_tracker_fixture.plot_prevented_saturation(start_target_date="2024-01-10",
                                                              end_target_date="2024-01-20")
    assert len(fig.data) == 1  # Only CA's second point ('2024-01-15', -10)
    ca_trace = fig.data[0]
    assert ca_trace.name == "CA"
    assert list(ca_trace.y) == [-10]


def test_plot_prevented_saturation_raises_on_no_paired_data_for_state(prevented_tracker_fixture):
    # TX only has a prediction, no actual for that pred_date/target_date/state combo
    with pytest.raises(KeyError):
        prevented_tracker_fixture.plot_prevented_saturation(state="TX")


def test_plot_prevented_saturation_raises_on_no_data_for_filters():
    pt = PreventedTracker()
    pt.register_prediction("2024-01-01", "2024-01-08", "CA", 200)
    pt.register_actual("2024-01-01", "2024-01-08", "CA", 150)
    with pytest.raises(ValueError, match="No data available for the selected filters."):
        pt.plot_prevented_saturation(state="FL")
    with pytest.raises(ValueError, match="No data available for the selected filters."):
        pt.plot_prevented_saturation(start_target_date="2030-01-01")


######################################################################################
#######                       SATISFACTION TRACKER                              ######
######################################################################################

@pytest.fixture
def satisfaction_tracker_fixture():
    st = SatisfactionTracker()
    # Week 1: Jan 1 - Jan 7, 2024
    st.register_satisfaction_score("2024-01-01", "CA", 5)
    st.register_satisfaction_score("2024-01-02", "CA", 4)  # CA Week 1 Avg: (5+4)/2 = 4.5
    st.register_satisfaction_score("2024-01-03", "NY", 3)  # NY Week 1 Avg: 3

    # Week 2: Jan 8 - Jan 14, 2024
    st.register_satisfaction_score("2024-01-08", "CA", 5)  # CA Week 2 Avg: 5
    st.register_satisfaction_score("2024-01-09", "NY", 2)
    st.register_satisfaction_score("2024-01-10", "NY", 4)  # NY Week 2 Avg: (2+4)/2 = 3
    return st


def test_satisfaction_tracker_initialization():
    st = SatisfactionTracker()
    assert st.data.empty
    assert list(st.data.columns) == ['date', 'state', 'score']
    assert st.data['date'].dtype == 'datetime64[ns, UTC]'
    assert st.data['score'].dtype == 'int64' or st.data['score'].dtype == 'int32'  # Pandas default int


def test_register_satisfaction_score(satisfaction_tracker_fixture):
    df = satisfaction_tracker_fixture.data
    assert len(df) == 6
    assert df.loc[0, 'state'] == "CA" and df.loc[0, 'score'] == 5
    assert df.loc[0, 'date'] == pd.Timestamp("2024-01-01", tz='UTC')
    assert df['date'].dtype == 'datetime64[ns, UTC]'


def test_register_satisfaction_score_invalid_score():
    st = SatisfactionTracker()
    with pytest.raises(ValueError, match="Satisfaction score must be an integer between 0 and 5."):
        st.register_satisfaction_score("2024-01-01", "CA", 6)
    with pytest.raises(ValueError, match="Satisfaction score must be an integer between 0 and 5."):
        st.register_satisfaction_score("2024-01-01", "CA", -1)
    with pytest.raises(ValueError, match="Satisfaction score must be an integer between 0 and 5."):
        st.register_satisfaction_score("2024-01-01", "CA", 3.5)


def test_plot_satisfaction_trends_default_weekly(satisfaction_tracker_fixture):
    fig = satisfaction_tracker_fixture.plot_satisfaction_trends()
    assert fig is not None
    assert len(fig.data) == 2  # CA and NY

    ca_trace = next(d for d in fig.data if d.name == 'CA')
    ny_trace = next(d for d in fig.data if d.name == 'NY')

    assert list(ca_trace.y) == [4.5, 5.0]

    assert list(ny_trace.y) == [3.0, 3.0]

    assert "Weekly Average Satisfaction Score" in fig.layout.title.text
    assert fig.layout.yaxis.title.text == "Average Satisfaction (0-5)"


def test_plot_satisfaction_trends_daily(satisfaction_tracker_fixture):
    fig = satisfaction_tracker_fixture.plot_satisfaction_trends(aggregation_period='D')
    assert "Daily Average Satisfaction Score" in fig.layout.title.text
    ca_trace = next(d for d in fig.data if d.name == 'CA')
    # CA daily: (2024-01-01, 5), (2024-01-02, 4), (2024-01-08, 5)
    assert len(ca_trace.x) == 3
    assert list(ca_trace.y) == [5.0, 4.0, 5.0]


def test_plot_satisfaction_trends_monthly(satisfaction_tracker_fixture):
    fig = satisfaction_tracker_fixture.plot_satisfaction_trends(aggregation_period='M')
    assert "Monthly Average Satisfaction Score" in fig.layout.title.text
    ca_trace = next(d for d in fig.data if d.name == 'CA')
    ny_trace = next(d for d in fig.data if d.name == 'NY')
    # All data is in Jan 2024. Month start is 2024-01-01
    # CA Jan Avg: (5+4+5)/3
    # NY Jan Avg: (3+2+4)/3
    assert len(ca_trace.x) == 1
    assert pytest.approx(ca_trace.y[0]) == (5 + 4 + 5) / 3
    assert pytest.approx(ny_trace.y[0]) == (3 + 2 + 4) / 3


def test_plot_satisfaction_trends_filtered_state(satisfaction_tracker_fixture):
    fig = satisfaction_tracker_fixture.plot_satisfaction_trends(state="CA", aggregation_period='W')
    assert len(fig.data) == 1
    assert fig.data[0].name == "CA"
    assert list(fig.data[0].y) == [4.5, 5.0]
    assert "for CA" in fig.layout.title.text


def test_plot_satisfaction_trends_filtered_date(satisfaction_tracker_fixture):
    # Only data from week 2 (Jan 8 onwards)
    fig = satisfaction_tracker_fixture.plot_satisfaction_trends(start_date="2024-01-08", aggregation_period='W')
    ca_trace = next(d for d in fig.data if d.name == 'CA')
    ny_trace = next(d for d in fig.data if d.name == 'NY')

    assert list(ca_trace.y) == [5.0]
    assert list(ny_trace.y) == [3.0]


def test_plot_satisfaction_trends_raises_on_empty_data():
    st = SatisfactionTracker()
    with pytest.raises(ValueError, match="No satisfaction data available for the selected filters."):
        st.plot_satisfaction_trends()
    st.register_satisfaction_score("2024-01-01", "CA", 5)
    with pytest.raises(ValueError, match="No satisfaction data available for the selected filters."):
        st.plot_satisfaction_trends(state="NY")