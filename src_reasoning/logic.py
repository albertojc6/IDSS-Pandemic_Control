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
        def configure_five_levels(var, ranges, prefixes):
            var[prefixes[0]] = fuzz.trimf(var.universe, ranges[0])
            var[prefixes[1]] = fuzz.trimf(var.universe, ranges[1])
            var[prefixes[2]] = fuzz.trimf(var.universe, ranges[2])
            var[prefixes[3]] = fuzz.trimf(var.universe, ranges[3])
            var[prefixes[4]] = fuzz.trimf(var.universe, ranges[4])

        # Define input and output variables
        self.ia = ctrl.Antecedent(np.arange(0, 10000, 1), 'ia')  # Incidència acumulada (per 100k)
        self.ocupacio = ctrl.Antecedent(np.arange(0, 101, 1), 'ocupacio')  # Ocupació hospitalària (%)
        self.mortalitat = ctrl.Antecedent(np.arange(0, 2000, 1), 'mortalitat')  # Taxa de mortalitat (per 100k)
        self.letalitat = ctrl.Antecedent(np.arange(0, 101, 1), 'letalitat')
        self.poblacio65 = ctrl.Antecedent(np.arange(0, 101, 1), 'poblacio65')  # % població >65 anys
        self.densitat = ctrl.Antecedent(np.arange(0, 15001, 1), 'densitat')  # Densitat (hab/km²)
        self.risc = ctrl.Consequent(np.arange(0, 101, 1), 'risc')  # Risc de confinament (0-100)

        configure_five_levels(self.ia, 
            [[0,0,25], [15,40,75], [60,100,140], [120,150,800], [170,250,10000]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])

        configure_five_levels(self.ocupacio,
            [[0,0,60], [50,60,70], [60,70,80], [70,80,90], [85,95,100]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])

        configure_five_levels(self.mortalitat,
            [[0,0,10], [5,8,12], [10,13,17], [15,18,20], [20,25,2000]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])

        configure_five_levels(self.letalitat,
            [[0,0,1.5], [1,2.5,4], [3,5,7], [5,6.5,8], [7,10,100]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])

        configure_five_levels(self.poblacio65,
            [[0,0,7.5], [5,7.5,12.5], [10,12.5,17.5], [15,17.5,22.5], [20,25,100]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])

        configure_five_levels(self.densitat,
            [[0,0,500], [300,800,1200], [1000,2000,3000], [2500,4000,6000], [5000,8000,15000]],
            ['Molt Baixa', 'Baixa', 'Moderada', 'Alta', 'Molt Alta'])


        # Risc
        self.risc['Molt Baix'] = fuzz.trimf(self.risc.universe, [0, 0, 20])
        self.risc['Baix'] = fuzz.trimf(self.risc.universe, [10, 25, 40])
        self.risc['Moderat'] = fuzz.trimf(self.risc.universe, [30, 45, 60])
        self.risc['Alt'] = fuzz.trimf(self.risc.universe, [50, 65, 80])
        self.risc['Molt Alt'] = fuzz.trimf(self.risc.universe, [70, 85, 95])
        self.risc['Extrem'] = fuzz.trimf(self.risc.universe, [90, 100, 100])

        rules = [
        # Regles per escenaris extremes
        ctrl.Rule(self.ia['Molt Alta'] & self.ocupacio['Molt Alta'], self.risc['Extrem']),
        ctrl.Rule(self.ocupacio['Molt Alta'] & (self.mortalitat['Molt Alta'] | self.letalitat['Molt Alta']), self.risc['Extrem']),
        ctrl.Rule(self.ia['Molt Alta'] & (self.densitat['Molt Alta']|self.densitat['Alta']), self.risc['Molt Alt']),

        # Combinacions altes de 3 factors
        ctrl.Rule(self.ia['Alta'] & self.ocupacio['Alta'] & self.densitat['Alta'], self.risc['Molt Alt']),
        ctrl.Rule(self.poblacio65['Molt Alta'] & self.mortalitat['Alta'] & self.letalitat['Alta'], self.risc['Molt Alt']),
        
        # Escenaris amb dos factors alts
        ctrl.Rule(self.ia['Alta'] & self.ocupacio['Moderada'], self.risc['Alt']),
        ctrl.Rule(self.mortalitat['Alta'] & self.poblacio65['Alta'], self.risc['Alt']),
        ctrl.Rule((self.densitat['Molt Alta']|self.densitat['Alta']) & self.ia['Moderada'], self.risc['Alt']),

        # Escenaris moderats amb compensacions
        ctrl.Rule(self.ia['Moderada'] & self.ocupacio['Baixa'] & self.densitat['Moderada'], self.risc['Moderat']),
        ctrl.Rule(self.mortalitat['Moderada'] & self.letalitat['Baixa'], self.risc['Moderat']),
        
        # Interaccions específiques
        ctrl.Rule((self.ia['Baixa'] | self.ocupacio['Baixa']) & self.poblacio65['Molt Alta'], self.risc['Moderat']),

        ctrl.Rule(self.densitat['Alta'] & self.letalitat['Moderada'], self.risc['Alt']),
        
        # Escenaris de risc baix controlat
        ctrl.Rule(self.ia['Baixa'] & self.ocupacio['Molt Baixa'] & self.mortalitat['Molt Baixa'], self.risc['Molt Baix']),
        ctrl.Rule(self.densitat['Baixa'] & self.letalitat['Molt Baixa'], self.risc['Baix']),
        
        # Regles de transició gradual
        ctrl.Rule(self.ia['Moderada'] & self.ocupacio['Baixa'] & self.mortalitat['Baixa'], self.risc['Baix']),
        ctrl.Rule(self.ia['Baixa'] & self.ocupacio['Moderada'] & self.letalitat['Moderada'], self.risc['Moderat']),

        # Combinacions amb població vulnerable
        ctrl.Rule(self.poblacio65['Alta'] & (self.ia['Baixa'] | self.mortalitat['Baixa']), self.risc['Moderat']),
        ctrl.Rule(self.poblacio65['Molt Alta'] & self.densitat['Alta'], self.risc['Alt']),

        ctrl.Rule(self.letalitat['Molt Alta'] & self.mortalitat['Moderada'], self.risc['Alt']),

        # Lletalitat elevada + Densitat baixa
        ctrl.Rule(self.letalitat['Molt Alta'] & self.densitat['Baixa'], self.risc['Moderat']),
        
        # Lletalitat elevada + Població vulnerable inexistent
        ctrl.Rule(self.letalitat['Molt Alta'] & self.poblacio65['Molt Baixa'], self.risc['Alt']),
        
        # Lletalitat elevada amb incidència/ocupació controlades
        ctrl.Rule(self.letalitat['Molt Alta'] & self.ia['Baixa'] & self.ocupacio['Molt Baixa'], self.risc['Moderat']),
        
        # Efecte acumulatiu de factors moderats
        ctrl.Rule((self.ia['Moderada'] | self.mortalitat['Moderada']) & (self.ocupacio['Moderada'] | self.letalitat['Moderada']), self.risc['Moderat']),
        
        # Regla residual ultra-específica
        ctrl.Rule((self.ia['Molt Baixa'] & self.ocupacio['Molt Baixa']) | (self.ocupacio['Molt Baixa'] & self.mortalitat['Molt Baixa']) | (self.ia['Molt Baixa'] & self.letalitat['Molt Baixa']),
            self.risc['Molt Baix'])
        ]
        self.risc_ctrl = ctrl.ControlSystem(rules)

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