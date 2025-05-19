import pandas as pd
from logic import FuzzyEpidemiology

def create_test_cases():
    # Crear instància de FuzzyEpidemiology
    fuzzy_system = FuzzyEpidemiology()

    # Dades demogràfiques base per a Artificial_High (New Mexico)
    base_demo = {
        'population_state': 2096829,  # New Mexico
        'pop_density_state': 17.27,
        'bedsTotal': 2.5,  
        'pop_0-9': 209682, 'pop_10-19': 209682, 'pop_20-29': 209682,
        'pop_30-39': 209682, 'pop_40-49': 209682, 'pop_50-59': 209682,
        'pop_60-69': 209682, 'pop_70-79': 125809, 'pop_80+': 125809  # 12% de >65
    }

    # Crear DataFrame replicant les dades demogràfiques pels 3 escenaris
    states = ['Artificial_High_No', 'Artificial_High_Selectiu', 'Artificial_High_Immediat']
    demographic_data = pd.DataFrame([base_demo] * 3, index=states).reset_index().rename(columns={'index': 'state'})

    # Dades de prediccions pels 3 escenaris
    predictions_data = pd.DataFrame({
        'state': states,
        'positiveIncrease': [500, 2000, 100000],  
        'hospitalizedIncrease': [0, 3000, 4500],  
        'deathIncrease': [10, 300, 2000]          
    })

    # Assignar dades artificials
    fuzzy_system.df_demografic = demographic_data
    fuzzy_system.df_predictions = predictions_data

    # Crear veïns artificials simplificats
    neighbor_data = pd.DataFrame({
        'state': states,
        'Artificial_High_No': [0, 0, 0],
        'Artificial_High_Selectiu': [0, 0, 0],
        'Artificial_High_Immediat': [0, 0, 0]
    })
    fuzzy_system.df_veins = neighbor_data
    fuzzy_system.veins_dict = {row['state']: [col for col in neighbor_data.columns if neighbor_data[col][i] == 1] 
                             for i, row in neighbor_data.iterrows()}

    # Executar i mostrar resultats amb gestió d'errors
    for state in states:
        result = fuzzy_system.get_knowledge(state)
        if 'error' in result:
            print(f"\nTest Case - State: {state} - Error: {result['error']}")
        else:
            print(f"\nTest Case - State: {state}")
            print(f"Input Metrics:")
            print(f"  IA: {result['metrics']['IA']:.2f}")
            print(f"  Ocupació (theta): {result['metrics']['theta']:.2f}")
            print(f"  Mortalitat (pi): {result['metrics']['pi']:.2f}")
            print(f"  Letalitat: {result['metrics']['letalitat']:.2f}")
            print(f"  Població >65: {result['metrics']['pop_>65']:.2f}")
            print(f"  Densitat: {result['metrics']['densitat']:.2f}")
            print(f"Output:")
            print(f"  Nivell de Risc: {result['metrics']['nivell_risc']:.2f}")
            print(f"  Confinement Recommendation: {result['recommendations']['confinament'][1]}")

if __name__ == "__main__":
    create_test_cases()