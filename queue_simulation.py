import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, norm
import time

class QueueSimulator:
    def __init__(self, arrival_rate, service_rate, num_customers=1000000):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_customers = num_customers
        self.arrival_times = []
        self.service_times = []
        self.departure_times = []
        self.waiting_times = []
        self.response_times = []
        self.server_occupation = 0

    def simulate(self):
        raise NotImplementedError("Subclasses must implement simulate()")

    def calculate_metrics(self):
        if len(self.response_times) == 0:
            return None
        
        avg_response_time = np.mean(self.response_times)
        avg_waiting_time = np.mean(self.waiting_times)
        server_utilization = self.server_occupation / self.departure_times[-1]
        
        # Calcul des valeurs théoriques pour M/M/1 (référence)
        rho = self.arrival_rate / self.service_rate
        if rho < 1:  # Condition de stabilité
            theoretical_response_time = 1 / (self.service_rate - self.arrival_rate)
            theoretical_waiting_time = rho / (self.service_rate - self.arrival_rate)
        else:
            theoretical_response_time = float('inf')
            theoretical_waiting_time = float('inf')
        
        return {
            'avg_response_time': avg_response_time,
            'avg_waiting_time': avg_waiting_time,
            'server_utilization': server_utilization,
            'theoretical_response_time': theoretical_response_time,
            'theoretical_waiting_time': theoretical_waiting_time,
            'rho': rho
        }

class MM1Queue(QueueSimulator):
    def simulate(self):
        print(f"  Simulation M/M/1 avec λ={self.arrival_rate}, μ={self.service_rate}")
        
        # Génération des temps d'arrivée (loi exponentielle)
        inter_arrival_times = expon.rvs(scale=1/self.arrival_rate, size=self.num_customers)
        self.arrival_times = np.cumsum(inter_arrival_times)
        
        # Génération des temps de service (loi exponentielle)
        self.service_times = expon.rvs(scale=1/self.service_rate, size=self.num_customers)
        
        # Simulation de la file d'attente
        self.departure_times = np.zeros(self.num_customers)
        self.waiting_times = np.zeros(self.num_customers)
        
        for i in range(self.num_customers):
            if i == 0:
                # Premier client : pas d'attente
                self.departure_times[i] = self.arrival_times[i] + self.service_times[i]
                self.waiting_times[i] = 0
            else:
                # Temps d'attente = max(0, temps de départ du client précédent - temps d'arrivée)
                self.waiting_times[i] = max(0, self.departure_times[i-1] - self.arrival_times[i])
                self.departure_times[i] = self.arrival_times[i] + self.waiting_times[i] + self.service_times[i]
        
        # Calcul des temps de réponse et occupation du serveur
        self.response_times = self.departure_times - self.arrival_times
        self.server_occupation = np.sum(self.service_times)

class GM1Queue(QueueSimulator):
    
    def simulate(self):
        print(f"  Simulation G/M/1 avec λ={self.arrival_rate}, μ={self.service_rate}")
        
        # Génération des temps d'arrivée avec une loi gamma
        # Paramètres choisis pour conserver le taux d'arrivée moyen λ
        shape = 2  # paramètre de forme (k=2 donne une variabilité modérée)
        scale = 1/(self.arrival_rate * shape)  # paramètre d'échelle pour avoir E[X] = 1/λ
        inter_arrival_times = gamma.rvs(shape, scale=scale, size=self.num_customers)
        self.arrival_times = np.cumsum(inter_arrival_times)
        
        # Génération des temps de service exponentiels (identique à M/M/1)
        self.service_times = expon.rvs(scale=1/self.service_rate, size=self.num_customers)
        
        # Simulation de la file d'attente (même logique que M/M/1)
        self.departure_times = np.zeros(self.num_customers)
        self.waiting_times = np.zeros(self.num_customers)
        
        for i in range(self.num_customers):
            if i == 0:
                self.departure_times[i] = self.arrival_times[i] + self.service_times[i]
                self.waiting_times[i] = 0
            else:
                self.waiting_times[i] = max(0, self.departure_times[i-1] - self.arrival_times[i])
                self.departure_times[i] = self.arrival_times[i] + self.waiting_times[i] + self.service_times[i]
        
        self.response_times = self.departure_times - self.arrival_times
        self.server_occupation = np.sum(self.service_times)

class MG1Queue(QueueSimulator):
    
    def simulate(self):
        print(f"  Simulation M/G/1 avec λ={self.arrival_rate}, μ={self.service_rate}")
        
        # Génération des temps d'arrivée exponentiels (identique à M/M/1)
        inter_arrival_times = expon.rvs(scale=1/self.arrival_rate, size=self.num_customers)
        self.arrival_times = np.cumsum(inter_arrival_times)
        
        # Génération des temps de service avec une loi normale tronquée
        mean = 1/self.service_rate  # temps de service moyen
        std_dev = mean/2  # écart-type (coefficient de variation = 0.5)
        # Utilisation de la valeur absolue pour éviter les temps négatifs
        self.service_times = np.abs(norm.rvs(loc=mean, scale=std_dev, size=self.num_customers))
        
        # Simulation de la file d'attente (même logique que M/M/1)
        self.departure_times = np.zeros(self.num_customers)
        self.waiting_times = np.zeros(self.num_customers)
        
        for i in range(self.num_customers):
            if i == 0:
                self.departure_times[i] = self.arrival_times[i] + self.service_times[i]
                self.waiting_times[i] = 0
            else:
                self.waiting_times[i] = max(0, self.departure_times[i-1] - self.arrival_times[i])
                self.departure_times[i] = self.arrival_times[i] + self.waiting_times[i] + self.service_times[i]
        
        self.response_times = self.departure_times - self.arrival_times
        self.server_occupation = np.sum(self.service_times)

def run_simulation(queue_type, arrival_rates, service_rate=1.0, num_customers=1000000, num_repetitions=5):
    results = {
        'arrival_rates': arrival_rates,
        'response_times': [],
        'waiting_times': [],
        'server_utilizations': [],
        'theoretical_response_times': [],
        'theoretical_waiting_times': [],
        'response_times_std': [],
        'waiting_times_std': []
    }
    
    print(f"\nSimulation {queue_type} en cours...")
    print(f"Paramètres: {num_customers} clients, {num_repetitions} répétitions par point")
    
    for i, arrival_rate in enumerate(arrival_rates):
        print(f"Point {i+1}/{len(arrival_rates)}: λ = {arrival_rate}")
        
        response_times = []
        waiting_times = []
        server_utilizations = []
        theoretical_response_times = []
        theoretical_waiting_times = []
        
        for rep in range(num_repetitions):
            print(f"    Répétition {rep+1}/{num_repetitions}")
            
            # Création du simulateur approprié
            if queue_type == 'MM1':
                simulator = MM1Queue(arrival_rate, service_rate, num_customers)
            elif queue_type == 'GM1':
                simulator = GM1Queue(arrival_rate, service_rate, num_customers)
            elif queue_type == 'MG1':
                simulator = MG1Queue(arrival_rate, service_rate, num_customers)
            else:
                raise ValueError(f"Type de file invalide: {queue_type}")
            
            # Exécution de la simulation
            start_time = time.time()
            simulator.simulate()
            end_time = time.time()
            print(f"      Temps d'exécution: {end_time - start_time:.2f}s")
            
            # Calcul des métriques
            metrics = simulator.calculate_metrics()
            
            response_times.append(metrics['avg_response_time'])
            waiting_times.append(metrics['avg_waiting_time'])
            server_utilizations.append(metrics['server_utilization'])
            theoretical_response_times.append(metrics['theoretical_response_time'])
            theoretical_waiting_times.append(metrics['theoretical_waiting_time'])
        
        # Calcul des moyennes et écarts-types
        results['response_times'].append(np.mean(response_times))
        results['waiting_times'].append(np.mean(waiting_times))
        results['server_utilizations'].append(np.mean(server_utilizations))
        results['theoretical_response_times'].append(np.mean(theoretical_response_times))
        results['theoretical_waiting_times'].append(np.mean(theoretical_waiting_times))
        results['response_times_std'].append(np.std(response_times))
        results['waiting_times_std'].append(np.std(waiting_times))
    
    return results

def plot_results(results_mm1, results_gm1, results_mg1):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison des modèles de files d\'attente M/M/1, G/M/1 et M/G/1', fontsize=16)
    
    arrival_rates = results_mm1['arrival_rates']
    
    # Temps de réponse moyen
    ax1 = axes[0, 0]
    ax1.plot(arrival_rates, results_mm1['response_times'], 'b-o', label='M/M/1 (simulation)', linewidth=2)
    ax1.plot(arrival_rates, results_mm1['theoretical_response_times'], 'b--', label='M/M/1 (théorique)', alpha=0.7)
    ax1.plot(arrival_rates, results_gm1['response_times'], 'r-s', label='G/M/1', linewidth=2)
    ax1.plot(arrival_rates, results_mg1['response_times'], 'g-^', label='M/G/1', linewidth=2)
    ax1.set_xlabel('Taux d\'arrivée (λ)')
    ax1.set_ylabel('Temps de réponse moyen')
    ax1.set_title('Temps de réponse moyen vs Taux d\'arrivée')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Temps d'attente moyen
    ax2 = axes[0, 1]
    ax2.plot(arrival_rates, results_mm1['waiting_times'], 'b-o', label='M/M/1 (simulation)', linewidth=2)
    ax2.plot(arrival_rates, results_mm1['theoretical_waiting_times'], 'b--', label='M/M/1 (théorique)', alpha=0.7)
    ax2.plot(arrival_rates, results_gm1['waiting_times'], 'r-s', label='G/M/1', linewidth=2)
    ax2.plot(arrival_rates, results_mg1['waiting_times'], 'g-^', label='M/G/1', linewidth=2)
    ax2.set_xlabel('Taux d\'arrivée (λ)')
    ax2.set_ylabel('Temps d\'attente moyen')
    ax2.set_title('Temps d\'attente moyen vs Taux d\'arrivée')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Taux d'utilisation du serveur
    ax3 = axes[1, 0]
    theoretical_utilization = arrival_rates  # ρ = λ/μ avec μ=1
    ax3.plot(arrival_rates, results_mm1['server_utilizations'], 'b-o', label='M/M/1', linewidth=2)
    ax3.plot(arrival_rates, results_gm1['server_utilizations'], 'r-s', label='G/M/1', linewidth=2)
    ax3.plot(arrival_rates, results_mg1['server_utilizations'], 'g-^', label='M/G/1', linewidth=2)
    ax3.plot(arrival_rates, theoretical_utilization, 'k--', label='Théorique (ρ=λ/μ)', alpha=0.7)
    ax3.set_xlabel('Taux d\'arrivée (λ)')
    ax3.set_ylabel('Taux d\'utilisation du serveur')
    ax3.set_title('Taux d\'utilisation du serveur vs Taux d\'arrivée')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Comparaison avec barres d'erreur pour λ=0.8
    ax4 = axes[1, 1]
    lambda_index = 7  # λ=0.8 correspond à l'index 7
    models = ['M/M/1', 'G/M/1', 'M/G/1']
    response_means = [results_mm1['response_times'][lambda_index], 
                     results_gm1['response_times'][lambda_index], 
                     results_mg1['response_times'][lambda_index]]
    response_stds = [results_mm1['response_times_std'][lambda_index], 
                    results_gm1['response_times_std'][lambda_index], 
                    results_mg1['response_times_std'][lambda_index]]
    
    x_pos = np.arange(len(models))
    bars = ax4.bar(x_pos, response_means, yerr=response_stds, capsize=5, 
                   color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_xlabel('Modèle de file d\'attente')
    ax4.set_ylabel('Temps de réponse moyen')
    ax4.set_title('Comparaison à λ=0.8 (avec intervalles de confiance)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for i, (bar, mean, std) in enumerate(zip(bars, response_means, response_stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('queue_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results_mm1, results_gm1, results_mg1):
    print("\n" + "="*80)
    print("TABLEAU DE SYNTHÈSE DES RÉSULTATS")
    print("="*80)
    print(f"{'λ':<6} {'M/M/1 TR':<10} {'G/M/1 TR':<10} {'M/G/1 TR':<10} {'M/M/1 TA':<10} {'G/M/1 TA':<10} {'M/G/1 TA':<10} {'ρ théo':<8}")
    print("-"*80)
    
    for i, lambda_val in enumerate(results_mm1['arrival_rates']):
        print(f"{lambda_val:<6.1f} "
              f"{results_mm1['response_times'][i]:<10.3f} "
              f"{results_gm1['response_times'][i]:<10.3f} "
              f"{results_mg1['response_times'][i]:<10.3f} "
              f"{results_mm1['waiting_times'][i]:<10.3f} "
              f"{results_gm1['waiting_times'][i]:<10.3f} "
              f"{results_mg1['waiting_times'][i]:<10.3f} "
              f"{lambda_val:<8.3f}")
    
    print("-"*80)
    print("TR = Temps de Réponse, TA = Temps d'Attente, ρ = Taux d'utilisation")
    print("="*80)

if __name__ == "__main__":
    # Paramètres de simulation
    arrival_rates = np.linspace(0.1, 0.9, 9)  # λ de 0.1 à 0.9 par pas de 0.1
    service_rate = 1.0  # μ = 1 (normalisé)
    num_customers = 1000000  # 1 million de clients par simulation
    num_repetitions = 5  # 5 répétitions pour la stabilité statistique
    
    print("="*60)
    print("SIMULATION COMPARATIVE DES FILES D'ATTENTE")
    print("M/M/1, G/M/1 et M/G/1")
    print("="*60)
    print(f"Paramètres:")
    print(f"- Taux de service μ = {service_rate}")
    print(f"- Taux d'arrivée λ ∈ [{arrival_rates[0]}, {arrival_rates[-1]}]")
    print(f"- Nombre de clients par simulation: {num_customers:,}")
    print(f"- Nombre de répétitions: {num_repetitions}")
    print("="*60)
    
    # Exécution des simulations
    start_total = time.time()
    
    results_mm1 = run_simulation('MM1', arrival_rates, service_rate, num_customers, num_repetitions)
    results_gm1 = run_simulation('GM1', arrival_rates, service_rate, num_customers, num_repetitions)
    results_mg1 = run_simulation('MG1', arrival_rates, service_rate, num_customers, num_repetitions)
    
    end_total = time.time()
    print(f"\nTemps total d'exécution: {end_total - start_total:.2f}s")
    
    # Affichage des résultats
    print_summary_table(results_mm1, results_gm1, results_mg1)
    
    # Génération des graphiques
    print("\nGénération des graphiques...")
    plot_results(results_mm1, results_gm1, results_mg1)
    
    print("\nSimulation terminée avec succès!")
    print("Les résultats ont été sauvegardés dans 'queue_simulation_results.png'")