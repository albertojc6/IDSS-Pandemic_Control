import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyEpidemiology:
    def __init__(self):
        """
        Initialize the FuzzyEpidemiology class with demographic, neighbor, and prediction data.
        
        Args:
            demographic_data_path (str): Path to the demographic data CSV file.
            neighbor_data_path (str): Path to the neighbor states CSV file.
            predictions_data_path (str): Path to the predictions data CSV file.
        """
        demographic_data_path="data/preprocessed/dataMatrix/static_stateMatrix.csv"
        neighbor_data_path="data/raw/neighbor_states/neighbor_states.csv"
        predictions_data_path="src_reasoning/predictions_matrix.csv"
        
        # Load demographic data
        columnes = [
            "state", "population_state", "pop_density_state", "bedsTotal",
            "pop_0-9", "pop_10-19", "pop_20-29", "pop_30-39",
            "pop_40-49", "pop_50-59", "pop_60-69", "pop_70-79", "pop_80+"
        ]
        df_demografic_original = pd.read_csv(demographic_data_path)
        self.df_demografic = df_demografic_original[columnes]

        # Load neighbor states data
        self.df_veins = pd.read_csv(neighbor_data_path)
        self.veins_dict = {}
        for idx, row in self.df_veins.iterrows():
            estat = row['state']
            veins = [vei for vei in self.df_veins.columns[1:] if row[vei] == 1]
            self.veins_dict[estat] = veins

        # Load predictions data
        self.df_predictions = pd.read_csv(predictions_data_path)

        # Initialize fuzzy logic system
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        """
        Set up the fuzzy logic system with antecedents, consequents, and rules.
        """
        # Define input and output variables
        self.ia = ctrl.Antecedent(np.arange(0, 10000, 1), 'ia')  # Incidència acumulada (per 100k)
        self.ocupacio = ctrl.Antecedent(np.arange(0, 101, 1), 'ocupacio')  # Ocupació hospitalària (%)
        self.mortalitat = ctrl.Antecedent(np.arange(0, 2000, 1), 'mortalitat')  # Taxa de mortalitat (per 100k)
        self.letalitat = ctrl.Antecedent(np.arange(0, 101, 1), 'letalitat')
        self.poblacio65 = ctrl.Antecedent(np.arange(0, 101, 1), 'poblacio65')  # % població >65 anys
        self.densitat = ctrl.Antecedent(np.arange(0, 15001, 1), 'densitat')  # Densitat (hab/km²)
        self.risc = ctrl.Consequent(np.arange(0, 101, 1), 'risc')  # Risc de confinament (0-100)

        # Membership functions
        # IA
        self.ia['Baixa'] = fuzz.trimf(self.ia.universe, [0, 0, 50])
        self.ia['Mitjana'] = fuzz.trimf(self.ia.universe, [30, 100, 150])
        self.ia['Alta'] = fuzz.trimf(self.ia.universe, [130, 200, 10000])

        # Ocupació Hospitalària
        self.ocupacio['Baixa'] = fuzz.trimf(self.ocupacio.universe, [0, 0, 70])
        self.ocupacio['Moderada'] = fuzz.trimf(self.ocupacio.universe, [60, 75, 80])
        self.ocupacio['Crítica'] = fuzz.trimf(self.ocupacio.universe, [75, 90, 100])

        # Mortalitat
        self.mortalitat['Baixa'] = fuzz.trimf(self.mortalitat.universe, [0, 0, 10])
        self.mortalitat['Mitjana'] = fuzz.trimf(self.mortalitat.universe, [5, 15, 25])
        self.mortalitat['Alta'] = fuzz.trimf(self.mortalitat.universe, [15, 25, 100])

        # Letalitat
        self.letalitat['Baixa'] = fuzz.trimf(self.letalitat.universe, [0, 0, 3])
        self.letalitat['Mitjana'] = fuzz.trimf(self.letalitat.universe, [1.5, 4, 6.5])
        self.letalitat['Alta'] = fuzz.trimf(self.letalitat.universe, [5, 10, 100])

        # Població >65
        self.poblacio65['Baixa'] = fuzz.trimf(self.poblacio65.universe, [0, 0, 10])
        self.poblacio65['Mitjana'] = fuzz.trimf(self.poblacio65.universe, [5, 15, 20])
        self.poblacio65['Alta'] = fuzz.trimf(self.poblacio65.universe, [18, 30, 30])

        # Densitat
        self.densitat['Baixa'] = fuzz.trimf(self.densitat.universe, [0, 0, 1500])
        self.densitat['Mitjana'] = fuzz.trimf(self.densitat.universe, [1000, 2000, 3000])
        self.densitat['Alta'] = fuzz.trimf(self.densitat.universe, [2500, 4000, 40000])

        # Risc
        self.risc['Molt Baix'] = fuzz.trimf(self.risc.universe, [0, 0, 20])
        self.risc['Baix'] = fuzz.trimf(self.risc.universe, [10, 25, 40])
        self.risc['Moderat'] = fuzz.trimf(self.risc.universe, [30, 45, 60])
        self.risc['Alt'] = fuzz.trimf(self.risc.universe, [50, 65, 80])
        self.risc['Molt Alt'] = fuzz.trimf(self.risc.universe, [70, 85, 95])
        self.risc['Extrem'] = fuzz.trimf(self.risc.universe, [90, 100, 100])

        # Define rules
        rule1 = ctrl.Rule(
            self.ia['Alta'] & self.ocupacio['Crítica'] & self.mortalitat['Alta'],
            self.risc['Extrem']
        )
        rule2 = ctrl.Rule(
            self.ia['Mitjana'] & self.ocupacio['Moderada'] & (self.poblacio65['Alta'] | self.densitat['Alta']) & 
            (self.letalitat['Baixa'] | self.letalitat['Mitjana']),
            self.risc['Alt']
        )
        rule3 = ctrl.Rule(
            self.ia['Baixa'] & self.ocupacio['Baixa'] & self.mortalitat['Baixa'] & self.letalitat['Baixa'],
            self.risc['Molt Baix']
        )
        rule4 = ctrl.Rule(
            (self.ia['Mitjana'] | self.mortalitat['Mitjana']) & self.densitat['Mitjana'] & self.letalitat['Baixa'],
            self.risc['Moderat']
        )
        rule5 = ctrl.Rule(
            self.ocupacio['Crítica'] & (self.poblacio65['Alta'] | self.mortalitat['Alta']),
            self.risc['Molt Alt']
        )
        rule6 = ctrl.Rule(
            self.ia['Alta'] & self.ocupacio['Moderada'] & self.mortalitat['Mitjana'],
            self.risc['Alt']
        )
        rule7 = ctrl.Rule(
            self.ia['Mitjana'] & (self.poblacio65['Mitjana'] | self.densitat['Mitjana']),
            self.risc['Moderat']
        )
        rule8 = ctrl.Rule(
            self.mortalitat['Alta'] & (self.ia['Baixa'] | self.ocupacio['Baixa']),
            self.risc['Alt']
        )
        rule9 = ctrl.Rule(
            self.densitat['Alta'] & self.ia['Mitjana'],
            self.risc['Alt']
        )
        rule10 = ctrl.Rule(
            self.poblacio65['Alta'] & self.letalitat['Alta'],
            self.risc['Alt']
        )
        rule11 = ctrl.Rule(
            self.ia['Alta'] & self.ocupacio['Baixa'] & self.letalitat['Baixa'],
            self.risc['Moderat']
        )
        rule12 = ctrl.Rule(
            self.ia['Baixa'] | self.ocupacio['Baixa'] | self.mortalitat['Baixa'],
            self.risc['Baix']
        )

        # Create control system
        self.risc_ctrl = ctrl.ControlSystem([
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
        # IA (Incidència Acumulada per 100k)
        row['IA'] = (row['positiveIncrease'] / row['population_state']) * 100000

        # θ (Ocupació hospitalària %)
        row['beds_real'] = (row['bedsTotal'] * row['population_state']) / 1000
        row['theta'] = (row['hospitalizedIncrease'] / row['beds_real']) * 100

        # π (Taxa de mortalitat per 100k habitants)
        row['pi'] = (row['deathIncrease'] / row['population_state']) * 100000

        # π (Taxa de letalitat %)
        row['letalitat'] = (row['deathIncrease'] / row['positiveIncrease']) * 100

        # % població >65
        row['pop_>65'] = row['pop_70-79'] + row['pop_80+']

        return row

    def _calculate_risk(self, row):
        """
        Calculate the risk level using the fuzzy logic system.
        
        Args:
            row (pd.Series): Row containing state data with metrics.
            
        Returns:
            float: Computed risk level.
        """
        risc_sim = ctrl.ControlSystemSimulation(self.risc_ctrl)
        inputs = {
            'ia': row['IA'],
            'ocupacio': row['theta'],
            'mortalitat': row['pi'],
            'letalitat': row['letalitat'],
            'poblacio65': row['pop_>65'],
            'densitat': row['pop_density_state']
        }
        for key, val in inputs.items():
            risc_sim.input[key] = val
        risc_sim.compute()
        return risc_sim.output['risc']

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
            return 'Selectiu'
        elif 60 <= risk_level < 80:
            return 'Estricte'
        else:
            return 'Immediat'

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
            return "No es necessita"

        veins = self.veins_dict.get(state, [])
        veins_disponibles = []
        for vei in veins:
            vei_data = self.combined_data[self.combined_data['state'] == vei]
            if not vei_data.empty:
                vei_pct = vei_data['beds_available_pct'].values[0]
                if vei_pct >= 30:
                    veins_disponibles.append(vei)

        if veins_disponibles:
            return f"Sí ➔ Veïns: {', '.join(veins_disponibles)}"
        return "No (cap veí vàlid)"

    def get_knowledge(self, state):
        """
        Process data for a single state and return predictions, metrics, and recommendations.
        
        Args:
            state (str): State name.
            
        Returns:
            dict: State data, metrics, and recommendations ([None, Valor] for beds, vaccination, confinement).
        """
        # Get prediction data for the state
        state_data = self.df_predictions[self.df_predictions['state'] == state].iloc[0]
        if pd.isna(state_data).any():
            return {"error": f"No prediction data found for state: {state}"}

        # Merge with demographic data
        self.combined_data = pd.merge(
            self.df_demografic,
            pd.DataFrame([state_data]),
            on="state",
            how="inner"
        )

        if self.combined_data.empty:
            return {"error": f"No demographic data found for state: {state}"}

        # Calculate metrics
        self.combined_data = self.combined_data.apply(self._calculate_metrics, axis=1)

        # Calculate risk level
        self.combined_data['nivell_risc'] = self.combined_data.apply(self._calculate_risk, axis=1)

        # Determine confinement
        risk_level = float(self.combined_data['nivell_risc'].iloc[0])
        confinement = self._determine_confinement(risk_level)

        # Calculate bed availability
        self.combined_data['bedsTotal2'] = (self.combined_data['bedsTotal'] / 1000) * self.combined_data['population_state']
        self.combined_data['bedsTotal3'] = self.combined_data['bedsTotal2'] - self.combined_data['hospitalizedIncrease']
        self.combined_data['beds_available_pct'] = (self.combined_data['bedsTotal3'] / self.combined_data['bedsTotal2']) * 100

        # Check for possible transfers
        beds_available_pct = float(self.combined_data['beds_available_pct'].iloc[0])
        beds_recommendation = self._check_transfers(state, beds_available_pct)

        # Vaccination percentage (simplified for a single state, assuming relative risk)
        vaccination_pct = round(risk_level, 2)

        # Prepare output with converted types
        state_data = {
            'state': state,
            'predictions': {
                'contagiats': int(self.combined_data['positiveIncrease'].iloc[0]),
                'hospitalitzats': int(self.combined_data['hospitalizedIncrease'].iloc[0]),
                'morts': int(self.combined_data['deathIncrease'].iloc[0])
            },
            'metrics': {
                'IA': round(float(self.combined_data['IA'].iloc[0]), 2),
                'theta': round(float(self.combined_data['theta'].iloc[0]), 2),
                'pi': round(float(self.combined_data['pi'].iloc[0]), 2),
                'letalitat': round(float(self.combined_data['letalitat'].iloc[0]), 2),
                'pop_>65': round(float(self.combined_data['pop_>65'].iloc[0]), 2),
                'densitat': round(float(self.combined_data['pop_density_state'].iloc[0]), 2),
                'nivell_risc': round(risk_level, 2)
            },
            'recommendations': {
                'llits': [None, beds_recommendation],
                'vacunació': [None, vaccination_pct],
                'confinament': [None, confinement]
            }
        }

        return state_data

# Example usage:
if __name__ == "__main__":
    # Initialize the class
    fuzzy_system = FuzzyEpidemiology()

    # Example call for a single state
    result = fuzzy_system.get_knowledge(state="California")
    print(result)