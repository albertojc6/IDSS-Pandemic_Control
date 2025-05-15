import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from app.models import StaticStateData, Recommendation, Prediction
from app.extensions import db
from datetime import date
from pathlib import Path
from sqlalchemy import desc

class FuzzyEpidemiology:
    def __init__(self):
        """
        Initialize the FuzzyEpidemiology class with demographic, neighbor, and prediction data.
        """
        # Load demographic data from database
        try:
            static_data = StaticStateData.query.all()
            self.df_demographic = pd.DataFrame([data.to_dict() for data in static_data])
        except Exception as e:
            print(f"Warning: Could not load static state data from database: {str(e)}")
            raise

        # Load neighbor states data
        base_path = Path(__file__).parent.parent.parent
        neighbor_data_path = base_path / "data" / "preprocessed" / "dataMatrix" / "neighbour_states.csv"
        self.df_neighbors = pd.read_csv(neighbor_data_path)
        self.neighbors_dict = {}
        for idx, row in self.df_neighbors.iterrows():
            state = row['state']
            neighbors = [neighbor for neighbor in self.df_neighbors.columns[1:] if row[neighbor] == 1]
            self.neighbors_dict[state] = neighbors

        # Initialize fuzzy logic system
        self._setup_fuzzy_system()
        
        # Initialize risk calculation cache
        self.risk_cache = {}  # Format: {state: {'prediction_id': id, 'risk_level': value}}

    def _setup_fuzzy_system(self):
        """
        Set up the fuzzy logic system with antecedents, consequents, and rules.
        """
        # Define input and output variables
        self.ia = ctrl.Antecedent(np.arange(0, 10000, 1), 'ia')  # Cumulative incidence (per 100k)
        self.occupancy = ctrl.Antecedent(np.arange(0, 101, 1), 'occupancy')  # Hospital occupancy (%)
        self.mortality = ctrl.Antecedent(np.arange(0, 2000, 1), 'mortality')  # Mortality rate (per 100k)
        self.lethality = ctrl.Antecedent(np.arange(0, 101, 1), 'lethality')
        self.population65 = ctrl.Antecedent(np.arange(0, 101, 1), 'population65')  # % population >65 years
        self.density = ctrl.Antecedent(np.arange(0, 15001, 1), 'density')  # Density (inhab/km²)
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')  # Confinement risk (0-100)

        # Membership functions
        # IA
        self.ia['Low'] = fuzz.trimf(self.ia.universe, [0, 0, 50])
        self.ia['Medium'] = fuzz.trimf(self.ia.universe, [30, 100, 150])
        self.ia['High'] = fuzz.trimf(self.ia.universe, [130, 200, 10000])

        # Hospital Occupancy
        self.occupancy['Low'] = fuzz.trimf(self.occupancy.universe, [0, 0, 70])
        self.occupancy['Moderate'] = fuzz.trimf(self.occupancy.universe, [60, 75, 80])
        self.occupancy['Critical'] = fuzz.trimf(self.occupancy.universe, [75, 90, 100])

        # Mortality
        self.mortality['Low'] = fuzz.trimf(self.mortality.universe, [0, 0, 10])
        self.mortality['Medium'] = fuzz.trimf(self.mortality.universe, [5, 15, 25])
        self.mortality['High'] = fuzz.trimf(self.mortality.universe, [15, 25, 100])

        # Lethality
        self.lethality['Low'] = fuzz.trimf(self.lethality.universe, [0, 0, 3])
        self.lethality['Medium'] = fuzz.trimf(self.lethality.universe, [1.5, 4, 6.5])
        self.lethality['High'] = fuzz.trimf(self.lethality.universe, [5, 10, 100])

        # Population >65
        self.population65['Low'] = fuzz.trimf(self.population65.universe, [0, 0, 10])
        self.population65['Medium'] = fuzz.trimf(self.population65.universe, [5, 15, 20])
        self.population65['High'] = fuzz.trimf(self.population65.universe, [18, 30, 30])

        # Density
        self.density['Low'] = fuzz.trimf(self.density.universe, [0, 0, 1500])
        self.density['Medium'] = fuzz.trimf(self.density.universe, [1000, 2000, 3000])
        self.density['High'] = fuzz.trimf(self.density.universe, [2500, 4000, 40000])

        # Risk
        self.risk['Very Low'] = fuzz.trimf(self.risk.universe, [0, 0, 20])
        self.risk['Low'] = fuzz.trimf(self.risk.universe, [10, 25, 40])
        self.risk['Moderate'] = fuzz.trimf(self.risk.universe, [30, 45, 60])
        self.risk['High'] = fuzz.trimf(self.risk.universe, [50, 65, 80])
        self.risk['Very High'] = fuzz.trimf(self.risk.universe, [70, 85, 95])
        self.risk['Extreme'] = fuzz.trimf(self.risk.universe, [90, 100, 100])

        # Define rules
        rule1 = ctrl.Rule(
            self.ia['High'] & self.occupancy['Critical'] & self.mortality['High'],
            self.risk['Extreme']
        )
        rule2 = ctrl.Rule(
            self.ia['Medium'] & self.occupancy['Moderate'] & (self.population65['High'] | self.density['High']) & 
            (self.lethality['Low'] | self.lethality['Medium']),
            self.risk['High']
        )
        rule3 = ctrl.Rule(
            self.ia['Low'] & self.occupancy['Low'] & self.mortality['Low'] & self.lethality['Low'],
            self.risk['Very Low']
        )
        rule4 = ctrl.Rule(
            (self.ia['Medium'] | self.mortality['Medium']) & self.density['Medium'] & self.lethality['Low'],
            self.risk['Moderate']
        )
        rule5 = ctrl.Rule(
            self.occupancy['Critical'] & (self.population65['High'] | self.mortality['High']),
            self.risk['Very High']
        )
        rule6 = ctrl.Rule(
            self.ia['High'] & self.occupancy['Moderate'] & self.mortality['Medium'],
            self.risk['High']
        )
        rule7 = ctrl.Rule(
            self.ia['Medium'] & (self.population65['Medium'] | self.density['Medium']),
            self.risk['Moderate']
        )
        rule8 = ctrl.Rule(
            self.mortality['High'] & (self.ia['Low'] | self.occupancy['Low']),
            self.risk['High']
        )
        rule9 = ctrl.Rule(
            self.density['High'] & self.ia['Medium'],
            self.risk['High']
        )
        rule10 = ctrl.Rule(
            self.population65['High'] & self.lethality['High'],
            self.risk['High']
        )
        rule11 = ctrl.Rule(
            self.ia['High'] & self.occupancy['Low'] & self.lethality['Low'],
            self.risk['Moderate']
        )
        rule12 = ctrl.Rule(
            self.ia['Low'] | self.occupancy['Low'] | self.mortality['Low'],
            self.risk['Low']
        )

        # Create control system
        self.risk_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6,
            rule7, rule8, rule9, rule10, rule11, rule12
        ])

    def _calculate_metrics(self, row):
        """
        Calculate epidemiological metrics for a given state row.
        
        Args:
            row (pd.Series): Row containing state data.
            
        Returns:
            pd.Series: Row with calculated metrics.
        """
        try:
            # Cumulative Incidence (per 100k)
            row['ia'] = (row['positiveIncrease'] / row['population_state']) * 100000 if row['population_state'] > 0 else 0

            # Hospital Occupancy (%)
            row['beds_real'] = (row['bedsTotal'] * row['population_state']) / 1000 if row['population_state'] > 0 else 0
            row['theta'] = (row['hospitalizedIncrease'] / row['beds_real']) * 100 if row['beds_real'] > 0 else 0

            # Mortality Rate (per 100k inhabitants)
            row['pi'] = (row['deathIncrease'] / row['population_state']) * 100000 if row['population_state'] > 0 else 0

            # Lethality Rate (%)
            row['lethality'] = (row['deathIncrease'] / row['positiveIncrease']) * 100 if row['positiveIncrease'] > 0 else 0

            # % population >65
            row['pop_>65'] = row['pop_70_79'] + row['pop_80_plus']

            # Ensure all metrics are finite numbers
            for metric in ['ia', 'theta', 'pi', 'lethality', 'pop_>65']:
                if not np.isfinite(row[metric]):
                    row[metric] = 0.0
                    print(f"Warning: Non-finite value for {metric} in {row['state']}, setting to 0")

            return row
        except Exception as e:
            print(f"Error calculating metrics for {row['state']}: {str(e)}")
            # Set default values for all metrics
            row['ia'] = 0.0
            row['theta'] = 0.0
            row['pi'] = 0.0
            row['lethality'] = 0.0
            row['pop_>65'] = row['pop_70_79'] + row['pop_80_plus']
            return row

    def _calculate_risk(self, row):
        """
        Calculate the risk level using the fuzzy logic system.
        
        Args:
            row (pd.Series): Row containing state data with metrics.
            
        Returns:
            float: Computed risk level.
        """
        risk_sim = ctrl.ControlSystemSimulation(self.risk_ctrl)
        inputs = {
            'ia': row['ia'],
            'occupancy': row['theta'],
            'mortality': row['pi'],
            'lethality': row['lethality'],
            'population65': row['pop_>65'],
            'density': row['pop_density_state']
        }
        for key, val in inputs.items():
            risk_sim.input[key] = val
        risk_sim.compute()
        return risk_sim.output['risk']

    def _determine_confinement(self, risk_level):
        """
        Determine confinement recommendation based on risk level.
        
        Args:
            risk_level (float): Computed risk level.
            
        Returns:
            str: Confinement recommendation.
        """
        if risk_level < 40:
            return 'No'
        elif 40 <= risk_level < 60:
            return 'Selective'
        elif 60 <= risk_level < 80:
            return 'Strict'
        else:
            return 'Immediate'

    def _check_transfers(self, state, beds_available_pct):
        """
        Check if patient transfers to neighboring states are possible.
        
        Args:
            state (str): State name.
            beds_available_pct (float): Percentage of available beds.
            
        Returns:
            str: Transfer recommendation.
        """
        if beds_available_pct >= 10:
            return "Not needed"

        neighbors = self.neighbors_dict.get(state, [])
        available_neighbors = []
        for neighbor in neighbors:
            neighbor_data = self.combined_data[self.combined_data['state'] == neighbor]
            if not neighbor_data.empty:
                neighbor_pct = neighbor_data['beds_available_pct'].values[0]
                if neighbor_pct >= 30:
                    available_neighbors.append(neighbor)

        if available_neighbors:
            return f"Yes ➔ Neighbors: {', '.join(available_neighbors)}"
        return "No (no valid neighbors)"

    def _get_cached_risk(self, state: str, prediction_id: int) -> float:
        """
        Get cached risk level for a state if the prediction hasn't changed.
        
        Args:
            state (str): State name
            prediction_id (int): ID of the current prediction
            
        Returns:
            float: Cached risk level if available, None otherwise
        """
        if state in self.risk_cache:
            cached_data = self.risk_cache[state]
            if cached_data['prediction_id'] == prediction_id:
                return cached_data['risk_level']
        return None

    def _cache_risk(self, state: str, prediction_id: int, risk_level: float):
        """
        Cache risk level for a state.
        
        Args:
            state (str): State name
            prediction_id (int): ID of the prediction
            risk_level (float): Calculated risk level
        """
        self.risk_cache[state] = {
            'prediction_id': prediction_id,
            'risk_level': risk_level
        }

    def get_knowledge(self, state: str, date: date) -> Recommendation:
        """
        Process data for a single state and date to generate recommendations.
        
        Args:
            state (str): State name.
            date (date): Date to generate recommendations for.
            
        Returns:
            Recommendation: Recommendation object with metrics and recommendations.
            
        Raises:
            ValueError: If no prediction data is found for the state and date.
        """
        # Get prediction data for the state and date from database
        prediction = Prediction.query.filter_by(
            state=state,
            date=date
        ).order_by(desc(Prediction.created_at)).first()
        
        if not prediction:
            raise ValueError(f"No prediction data found for state {state} on {date}")

        print(f"\nUsing prediction for {state} on {date}:")
        print(f"  - Prediction ID: {prediction.id}")
        print(f"  - Created at: {prediction.created_at}")
        print(f"  - Positive Increase Sum: {prediction.positive_increase_sum}")
        print(f"  - Hospitalized Increase Sum: {prediction.hospitalized_increase_sum}")
        print(f"  - Death Increase Sum: {prediction.death_increase_sum}")

        # Convert prediction to DataFrame format
        prediction_dict = prediction.to_dict()
        state_data = pd.DataFrame([{
            'state': state,
            'positiveIncrease': prediction_dict['positive_increase_sum'],
            'hospitalizedIncrease': prediction_dict['hospitalized_increase_sum'],
            'deathIncrease': prediction_dict['death_increase_sum']
        }])

        # Merge with demographic data
        self.combined_data = pd.merge(
            self.df_demographic,
            state_data,
            on="state",
            how="inner"
        )

        if self.combined_data.empty:
            raise ValueError(f"No demographic data found for state: {state}")

        # Calculate metrics
        self.combined_data = self.combined_data.apply(self._calculate_metrics, axis=1)

        # Calculate risk level
        self.combined_data['risk_level'] = self.combined_data.apply(self._calculate_risk, axis=1)

        # Get risk levels for all states
        all_states_risk = []
        for state_name in self.df_demographic['state'].unique():
            try:
                # For the current state, use the exact date
                # For other states, get their latest prediction
                if state_name == state:
                    state_prediction = Prediction.query.filter_by(
                        state=state_name,
                        date=date
                    ).order_by(desc(Prediction.created_at)).first()
                else:
                    state_prediction = Prediction.query.filter_by(
                        state=state_name
                    ).order_by(desc(Prediction.date), desc(Prediction.created_at)).first()
                
                if state_prediction:
                    # Check cache first
                    cached_risk = self._get_cached_risk(state_name, state_prediction.id)
                    if cached_risk is not None:
                        all_states_risk.append({
                            'state': state_name,
                            'risk_level': cached_risk
                        })
                        continue

                    # Get demographic data for this state
                    state_demographic = self.df_demographic[self.df_demographic['state'] == state_name]
                    
                    if not state_demographic.empty:
                        # Create state data with predictions
                        state_data = pd.DataFrame([{
                            'state': state_name,
                            'positiveIncrease': state_prediction.positive_increase_sum,
                            'hospitalizedIncrease': state_prediction.hospitalized_increase_sum,
                            'deathIncrease': state_prediction.death_increase_sum
                        }])
                        
                        # Merge with demographic data
                        state_combined = pd.merge(
                            state_demographic,
                            state_data,
                            on="state",
                            how="inner"
                        )
                        
                        if not state_combined.empty:
                            # Calculate metrics and risk level
                            state_combined = state_combined.apply(self._calculate_metrics, axis=1)
                            state_combined['risk_level'] = state_combined.apply(self._calculate_risk, axis=1)
                            
                            risk_level = float(state_combined['risk_level'].iloc[0])
                            
                            # Cache the result
                            self._cache_risk(state_name, state_prediction.id, risk_level)
                            
                            all_states_risk.append({
                                'state': state_name,
                                'risk_level': risk_level
                            })
            except Exception as e:
                print(f"Error calculating risk for {state_name}: {str(e)}")
                continue

        if not all_states_risk:
            raise ValueError("No risk levels could be calculated for any state")

        # Convert to DataFrame and calculate vaccination percentages
        df_risk = pd.DataFrame(all_states_risk)
        df_risk = df_risk.sort_values(by='risk_level', ascending=False)
        total_risk = df_risk['risk_level'].sum()
        
        # Calculate vaccination percentage for each state
        df_risk['vaccination_pct'] = (df_risk['risk_level'] / total_risk * 100).round(2)
        
        # Adjust for rounding errors to ensure total is exactly 100%
        diff = 100 - df_risk['vaccination_pct'].sum()
        if diff != 0:
            idx_max = df_risk['vaccination_pct'].idxmax()
            df_risk.at[idx_max, 'vaccination_pct'] += diff

        # Get vaccination percentage for current state
        state_vaccination_pct = float(df_risk[df_risk['state'] == state]['vaccination_pct'].iloc[0])

        # Determine confinement
        risk_level = float(self.combined_data['risk_level'].iloc[0])
        confinement = self._determine_confinement(risk_level)

        # Calculate bed availability
        self.combined_data['bedsTotal2'] = (self.combined_data['bedsTotal'] / 1000) * self.combined_data['population_state']
        self.combined_data['bedsTotal3'] = self.combined_data['bedsTotal2'] - self.combined_data['hospitalizedIncrease']
        self.combined_data['beds_available_pct'] = (self.combined_data['bedsTotal3'] / self.combined_data['bedsTotal2']) * 100

        # Check for possible transfers
        beds_available_pct = float(self.combined_data['beds_available_pct'].iloc[0])
        beds_recommendation = self._check_transfers(state, beds_available_pct)

        # Create and store recommendation
        try:
            recommendation = Recommendation(
                state=state,
                date=date,
                infected=int(self.combined_data['positiveIncrease'].iloc[0]),
                hospitalized=int(self.combined_data['hospitalizedIncrease'].iloc[0]),
                deaths=int(self.combined_data['deathIncrease'].iloc[0]),
                ia=round(float(self.combined_data['ia'].iloc[0]), 2),
                theta=round(float(self.combined_data['theta'].iloc[0]), 2),
                pi=round(float(self.combined_data['pi'].iloc[0]), 2),
                lethality=round(float(self.combined_data['lethality'].iloc[0]), 2),
                pop_over_65=round(float(self.combined_data['pop_>65'].iloc[0]), 2),
                density=round(float(self.combined_data['pop_density_state'].iloc[0]), 2),
                risk_level=round(risk_level, 2),
                beds_recommendation=beds_recommendation,
                vaccination_percentage=state_vaccination_pct,
                confinement_level=confinement
            )
            db.session.add(recommendation)
            db.session.commit()
            return recommendation
        except Exception as e:
            print(f"Warning: Could not save recommendation to database: {str(e)}")
            db.session.rollback()
            raise

# Example usage:
if __name__ == "__main__":
    # Initialize the class
    fuzzy_system = FuzzyEpidemiology()

    # Example call for a single state
    result = fuzzy_system.get_knowledge(state="California", date=date.today())
    print(result)