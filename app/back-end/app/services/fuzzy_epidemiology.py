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
        def configure_five_levels(var, ranges, prefixes):
            var[prefixes[0]] = fuzz.trimf(var.universe, ranges[0])
            var[prefixes[1]] = fuzz.trimf(var.universe, ranges[1])
            var[prefixes[2]] = fuzz.trimf(var.universe, ranges[2])
            var[prefixes[3]] = fuzz.trimf(var.universe, ranges[3])
            var[prefixes[4]] = fuzz.trimf(var.universe, ranges[4])

        # Define input and output variables
        self.ia = ctrl.Antecedent(np.arange(0, 10000, 1), 'ia')  # Cumulative incidence (per 100k)
        self.occupancy = ctrl.Antecedent(np.arange(0, 101, 1), 'occupancy')  # Hospital occupancy (%)
        self.mortality = ctrl.Antecedent(np.arange(0, 2000, 1), 'mortality')  # Mortality rate (per 100k)
        self.lethality = ctrl.Antecedent(np.arange(0, 101, 1), 'lethality')
        self.population65 = ctrl.Antecedent(np.arange(0, 101, 1), 'population65')  # % population >65 years
        self.density = ctrl.Antecedent(np.arange(0, 15001, 1), 'density')  # Density (inhab/km²)
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')  # Confinement risk (0-100)

        configure_five_levels(self.ia, 
            [[0,0,25], [15,40,75], [60,100,140], [120,150,800], [170,250,10000]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        configure_five_levels(self.occupancy,
            [[0,0,60], [50,60,70], [60,70,80], [70,80,90], [85,95,100]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        configure_five_levels(self.mortality,
            [[0,0,10], [5,8,12], [10,13,17], [15,18,20], [20,25,2000]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        configure_five_levels(self.lethality,
            [[0,0,1.5], [1,2.5,4], [3,5,7], [5,6.5,8], [7,10,100]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        configure_five_levels(self.population65,
            [[0,0,7.5], [5,7.5,12.5], [10,12.5,17.5], [15,17.5,22.5], [20,25,100]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        configure_five_levels(self.density,
            [[0,0,500], [300,800,1200], [1000,2000,3000], [2500,4000,6000], [5000,8000,15000]],
            ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

        # Risk
        self.risk['Very Low'] = fuzz.trimf(self.risk.universe, [0, 0, 20])
        self.risk['Low'] = fuzz.trimf(self.risk.universe, [10, 25, 40])
        self.risk['Moderate'] = fuzz.trimf(self.risk.universe, [30, 45, 60])
        self.risk['High'] = fuzz.trimf(self.risk.universe, [50, 65, 80])
        self.risk['Very High'] = fuzz.trimf(self.risk.universe, [70, 85, 95])
        self.risk['Extreme'] = fuzz.trimf(self.risk.universe, [90, 100, 100])

        rules = [
            # Rules for extreme scenarios
            ctrl.Rule(self.ia['Very High'] & self.occupancy['Very High'], self.risk['Extreme']),
            ctrl.Rule(self.occupancy['Very High'] & (self.mortality['Very High'] | self.lethality['Very High']), self.risk['Extreme']),
            ctrl.Rule(self.ia['Very High'] & (self.density['Very High']|self.density['High']), self.risk['Very High']),

            # High combinations of 3 factors
            ctrl.Rule(self.ia['High'] & self.occupancy['High'] & self.density['High'], self.risk['Very High']),
            ctrl.Rule(self.population65['Very High'] & self.mortality['High'] & self.lethality['High'], self.risk['Very High']),
            
            # Scenarios with two high factors
            ctrl.Rule(self.ia['High'] & self.occupancy['Moderate'], self.risk['High']),
            ctrl.Rule(self.mortality['High'] & self.population65['High'], self.risk['High']),
            ctrl.Rule((self.density['Very High']|self.density['High']) & self.ia['Moderate'], self.risk['High']),

            # Moderate scenarios with compensations
            ctrl.Rule(self.ia['Moderate'] & self.occupancy['Low'] & self.density['Moderate'], self.risk['Moderate']),
            ctrl.Rule(self.mortality['Moderate'] & self.lethality['Low'], self.risk['Moderate']),
            
            # Specific interactions
            ctrl.Rule((self.ia['Low'] | self.occupancy['Low']) & self.population65['Very High'], self.risk['Moderate']),

            ctrl.Rule(self.density['High'] & self.lethality['Moderate'], self.risk['High']),
            
            # Controlled low risk scenarios
            ctrl.Rule(self.ia['Low'] & self.occupancy['Very Low'] & self.mortality['Very Low'], self.risk['Very Low']),
            ctrl.Rule(self.density['Low'] & self.lethality['Very Low'], self.risk['Low']),
            
            # Gradual transition rules
            ctrl.Rule(self.ia['Moderate'] & self.occupancy['Low'] & self.mortality['Low'], self.risk['Low']),
            ctrl.Rule(self.ia['Low'] & self.occupancy['Moderate'] & self.lethality['Moderate'], self.risk['Moderate']),

            # Combinations with vulnerable population
            ctrl.Rule(self.population65['High'] & (self.ia['Low'] | self.mortality['Low']), self.risk['Moderate']),
            ctrl.Rule(self.population65['Very High'] & self.density['High'], self.risk['High']),

            ctrl.Rule(self.lethality['Very High'] & self.mortality['Moderate'], self.risk['High']),

            # High lethality + Low density
            ctrl.Rule(self.lethality['Very High'] & self.density['Low'], self.risk['Moderate']),
            
            # High lethality + No vulnerable population
            ctrl.Rule(self.lethality['Very High'] & self.population65['Very Low'], self.risk['High']),
            
            # High lethality with controlled incidence/occupancy
            ctrl.Rule(self.lethality['Very High'] & self.ia['Low'] & self.occupancy['Very Low'], self.risk['Moderate']),
            
            # Cumulative effect of moderate factors
            ctrl.Rule((self.ia['Moderate'] | self.mortality['Moderate']) & (self.occupancy['Moderate'] | self.lethality['Moderate']), self.risk['Moderate']),
            
            # Ultra-specific residual rule
            ctrl.Rule((self.ia['Very Low'] & self.occupancy['Very Low']) | (self.occupancy['Very Low'] & self.mortality['Very Low']) | (self.ia['Very Low'] & self.lethality['Very Low']),
                self.risk['Very Low'])
        ]
        self.risk_ctrl = ctrl.ControlSystem(rules)

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
        if beds_available_pct >= 50:
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

    def _recalculate_vaccination_percentages(self, date: date):
        """
        Recalcula els percentatges de vacunació per a tots els estats basant-se en el risc total.
        
        Args:
            date (date): Data per la qual recalcular els percentatges
            
        Returns:
            dict: Diccionari amb els nous percentatges de vacunació per cada estat
        """
        all_states_risk = []
        
        # Obtenir el risc per a tots els estats
        for state_name in self.df_demographic['state'].unique():
            try:
                # Obtenir la predicció més recent per a cada estat
                state_prediction = Prediction.query.filter_by(
                    state=state_name,
                    date=date
                ).order_by(desc(Prediction.created_at)).first()
                
                if not state_prediction:
                    continue
                
                # Obtenir dades demogràfiques per a aquest estat
                state_demographic = self.df_demographic[self.df_demographic['state'] == state_name]
                
                if not state_demographic.empty:
                    # Crear dades de l'estat amb prediccions
                    state_data = pd.DataFrame([{
                        'state': state_name,
                        'positiveIncrease': state_prediction.positive_increase_sum,
                        'hospitalizedIncrease': state_prediction.hospitalized_increase_sum,
                        'deathIncrease': state_prediction.death_increase_sum
                    }])
                    
                    # Combinar amb dades demogràfiques
                    state_combined = pd.merge(
                        state_demographic,
                        state_data,
                        on="state",
                        how="inner"
                    )
                    
                    if not state_combined.empty:
                        # Calcular mètriques i nivell de risc
                        state_combined = state_combined.apply(self._calculate_metrics, axis=1)
                        state_combined['risk_level'] = state_combined.apply(self._calculate_risk, axis=1)
                        
                        risk_level = float(state_combined['risk_level'].iloc[0])
                        all_states_risk.append({
                            'state': state_name,
                            'risk_level': risk_level
                        })
            except Exception as e:
                print(f"Error calculating risk for {state_name}: {str(e)}")
                continue
        
        if not all_states_risk:
            raise ValueError("No risk levels could be calculated for any state")
        
        # Convertir a DataFrame i calcular percentatges de vacunació
        df_risk = pd.DataFrame(all_states_risk)
        df_risk = df_risk.sort_values(by='risk_level', ascending=False)
        
        # Calcular risc total
        total_risk = df_risk['risk_level'].sum()
        
        # Calcular percentatge de vacunació per cada estat
        df_risk['vaccination_pct'] = (df_risk['risk_level'] / total_risk * 100).round(2)
        
        # Ajustar per errors d'arrodoniment
        diff = 100 - df_risk['vaccination_pct'].sum()
        if diff != 0:
            idx_max = df_risk['vaccination_pct'].idxmax()
            df_risk.at[idx_max, 'vaccination_pct'] += diff
        
        # Convertir a diccionari per fàcil accés
        vaccination_dict = dict(zip(df_risk['state'], df_risk['vaccination_pct']))
        
        return vaccination_dict

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

        # Recalcular percentatges de vacunació per a tots els estats
        vaccination_percentages = self._recalculate_vaccination_percentages(date)
        
        # Obtenir el percentatge de vacunació per a l'estat actual
        state_vaccination_pct = vaccination_percentages.get(state, 0.0)

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