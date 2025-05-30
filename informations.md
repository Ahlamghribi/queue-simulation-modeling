# Simulateur de Files d'Attente - Comparaison M/M/1, G/M/1 et M/G/1

Ce projet implémente un simulateur complet pour analyser et comparer les performances de trois modèles classiques de files d'attente mono-serveur : M/M/1, G/M/1 et M/G/1.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Modèles implémentés](#modèles-implémentés)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture du code](#architecture-du-code)
- [Métriques calculées](#métriques-calculées)
- [Résultats et visualisations](#résultats-et-visualisations)
- [Paramètres de simulation](#paramètres-de-simulation)
- [Interprétation des résultats](#interprétation-des-résultats)
- [Contribution](#contribution)

## 🎯 Vue d'ensemble

Ce simulateur permet d'étudier le comportement des files d'attente en faisant varier les distributions des temps d'arrivée et de service. Il compare les performances théoriques et simulées, offrant une analyse approfondie des métriques de performance comme les temps d'attente, temps de réponse et taux d'utilisation du serveur.

### Objectifs principaux
- Comparer les performances des modèles M/M/1, G/M/1 et M/G/1
- Valider les résultats théoriques par simulation Monte Carlo
- Analyser l'impact de la variabilité des distributions sur les performances
- Fournir des visualisations claires pour l'interprétation des résultats

## 📊 Modèles implémentés

### M/M/1 (Markovien/Markovien/1 serveur)
- **Arrivées** : Processus de Poisson (loi exponentielle)
- **Service** : Temps de service exponentiels
- **Caractéristiques** : Modèle de référence avec formules analytiques connues

### G/M/1 (Général/Markovien/1 serveur)
- **Arrivées** : Distribution générale (loi Gamma avec paramètre de forme k=2)
- **Service** : Temps de service exponentiels
- **Caractéristiques** : Variabilité réduite des arrivées par rapport à M/M/1

### M/G/1 (Markovien/Général/1 serveur)
- **Arrivées** : Processus de Poisson (loi exponentielle)
- **Service** : Distribution générale (loi normale tronquée)
- **Caractéristiques** : Variabilité contrôlée des temps de service

## 🛠 Installation

### Prérequis
```bash
Python >= 3.7
```

### Dépendances
```bash
pip install numpy matplotlib scipy
```

### Installation complète
```bash
# Cloner le repository
git clone <url-du-repository>
cd queue-simulation

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Exécution basique
```bash
python queue_simulator.py
```

### Personnalisation des paramètres
```python
# Modifier les paramètres dans le script principal
arrival_rates = np.linspace(0.1, 0.9, 9)  # Taux d'arrivée de 0.1 à 0.9
service_rate = 1.0                         # Taux de service normalisé
num_customers = 1000000                    # Nombre de clients par simulation
num_repetitions = 5                        # Répétitions pour stabilité statistique
```

### Exemple d'utilisation programmatique
```python
from queue_simulator import MM1Queue, GM1Queue, MG1Queue

# Créer un simulateur M/M/1
simulator = MM1Queue(arrival_rate=0.8, service_rate=1.0, num_customers=100000)

# Exécuter la simulation
simulator.simulate()

# Calculer les métriques
metrics = simulator.calculate_metrics()
print(f"Temps de réponse moyen: {metrics['avg_response_time']:.3f}")
print(f"Temps d'attente moyen: {metrics['avg_waiting_time']:.3f}")
print(f"Utilisation du serveur: {metrics['server_utilization']:.3f}")
```

## 🏗 Architecture du code

### Structure des classes

```
QueueSimulator (classe abstraite)
├── MM1Queue
├── GM1Queue
└── MG1Queue
```

### Classe QueueSimulator
**Attributs principaux :**
- `arrival_rate` : Taux d'arrivée λ
- `service_rate` : Taux de service μ
- `num_customers` : Nombre de clients à simuler
- `arrival_times` : Temps d'arrivée de chaque client
- `service_times` : Temps de service de chaque client
- `waiting_times` : Temps d'attente de chaque client
- `response_times` : Temps de réponse total

**Méthodes principales :**
- `simulate()` : Méthode abstraite pour la simulation
- `calculate_metrics()` : Calcul des métriques de performance

### Fonctions utilitaires
- `run_simulation()` : Exécute une série de simulations avec répétitions
- `plot_results()` : Génère les graphiques de comparaison
- `print_summary_table()` : Affiche un tableau de synthèse

## 📈 Métriques calculées

### Métriques de performance
- **Temps de réponse moyen** : Temps total passé par un client dans le système
- **Temps d'attente moyen** : Temps passé en file d'attente avant le service
- **Taux d'utilisation du serveur** : Fraction de temps où le serveur est occupé

### Comparaisons théoriques (M/M/1)
- **Temps de réponse théorique** : `1/(μ-λ)`
- **Temps d'attente théorique** : `ρ/(μ-λ)` où `ρ = λ/μ`
- **Condition de stabilité** : `ρ < 1`

### Statistiques
- Moyennes sur plusieurs répétitions
- Écarts-types pour évaluer la variabilité
- Intervalles de confiance

## 📊 Résultats et visualisations

Le programme génère automatiquement :

### Graphique 1 : Temps de réponse moyen vs Taux d'arrivée
- Comparaison des trois modèles
- Validation théorique pour M/M/1
- Mise en évidence des différences de performance

### Graphique 2 : Temps d'attente moyen vs Taux d'arrivée
- Évolution non-linéaire près de la saturation
- Impact de la variabilité des distributions

### Graphique 3 : Taux d'utilisation du serveur
- Vérification de la relation `ρ = λ/μ`
- Convergence vers la théorie

### Graphique 4 : Comparaison détaillée à λ=0.8
- Barres d'erreur pour les intervalles de confiance
- Analyse statistique de la variabilité

### Tableau de synthèse
Tableau récapitulatif avec toutes les métriques pour chaque valeur de λ.

## ⚙️ Paramètres de simulation

### Paramètres par défaut
```python
arrival_rates = np.linspace(0.1, 0.9, 9)  # 9 points de λ
service_rate = 1.0                         # μ normalisé
num_customers = 1000000                    # 1M clients/simulation
num_repetitions = 5                        # 5 répétitions/point
```

### Recommandations
- **num_customers** : Min 100,000 pour la stabilité statistique
- **num_repetitions** : Min 3-5 pour évaluer la variabilité
- **arrival_rates** : Éviter λ > 0.95 (temps de calcul exponentiel)

### Distribution des temps de service
- **G/M/1** : Gamma(k=2) pour variabilité modérée
- **M/G/1** : Normale tronquée (CV=0.5) pour réalisme

## 🔍 Interprétation des résultats

### Observations typiques

1. **M/M/1** : Référence théorique
   - Correspond exactement aux formules analytiques
   - Variabilité maximale (coefficient de variation = 1)

2. **G/M/1** : Arrivées plus régulières
   - Performances généralement meilleures que M/M/1
   - Temps d'attente réduits grâce à la régularité des arrivées

3. **M/G/1** : Services plus prévisibles
   - Performances variables selon le coefficient de variation
   - Impact de la formule de Pollaczek-Khinchine

### Facteurs d'influence
- **Taux d'utilisation ρ** : Performance dégradée près de ρ=1
- **Variabilité** : Plus de variabilité = performances dégradées
- **Type de distribution** : Impact asymétrique sur arrivées vs services

## 📁 Structure des fichiers

```
queue-simulation/
├── queue_simulator.py          # Code principal
├── informations.md                   # Ce fichier
├── requirements.txt            # Dépendances Python
├── queue_simulation_results.png # Graphiques générés
```

## 🧪 Tests et validation

### Validation M/M/1
Le modèle M/M/1 est validé contre les formules théoriques :
- Erreur relative < 1% pour 1M+ clients
- Convergence vérifiée sur multiple répétitions

### Tests de cohérence
- Conservation du flux : arrivées = départs
- Monotonie : métriques croissantes avec ρ
- Limites : comportement correct quand ρ→1

## 🤝 Contribution

### Comment contribuer
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

### Améliorations suggérées
- Ajout de nouveaux modèles (M/M/c, G/G/1)
- Interface graphique interactive
- Export des résultats en formats divers
- Optimisation des performances de calcul
- Tests unitaires automatisés

## 🙏 Remerciements

- Théorie des files d'attente : Kendall, Little, Pollaczek, Khinchine
- Bibliothèques Python : NumPy, SciPy, Matplotlib
- Communauté open source pour les retours et améliorations 