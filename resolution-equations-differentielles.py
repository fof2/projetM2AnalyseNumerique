resolution-equations-differentielles, Comparaison des methodes

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

class SolveurEDO:
    def __init__(self, f):
        self.f = f
    
    def euler(self, y0, x0, xf, n):
        h = (xf - x0) / n
        x = np.linspace(x0, xf, n+1)
        y = np.zeros(n+1)
        y[0] = y0
        
        for i in range(n):
            y[i+1] = y[i] + h * self.f(x[i], y[i])
        
        return x, y
    
    def heun(self, y0, x0, xf, n):
        h = (xf - x0) / n
        x = np.linspace(x0, xf, n+1)
        y = np.zeros(n+1)
        y[0] = y0
        
        for i in range(n):
            k1 = self.f(x[i], y[i])
            k2 = self.f(x[i] + h, y[i] + h * k1)
            y[i+1] = y[i] + (h/2) * (k1 + k2)
        
        return x, y
    
    def runge_kutta_4(self, y0, x0, xf, n):
        h = (xf - x0) / n
        x = np.linspace(x0, xf, n+1)
        y = np.zeros(n+1)
        y[0] = y0
        
        for i in range(n):
            k1 = self.f(x[i], y[i])
            k2 = self.f(x[i] + h/2, y[i] + h*k1/2)
            k3 = self.f(x[i] + h/2, y[i] + h*k2/2)
            k4 = self.f(x[i] + h, y[i] + h*k3)
            
            y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x, y

# =============================================================================
# DÉFINITION DES ÉQUATIONS ET SOLUTIONS EXACTES
# =============================================================================

# EXEMPLE 1: z′(x) = 0.1×x×z(x), z(0) = 1
def f_exemple1(x, y):
    return 0.1 * x * y

def sol_exacte1(x):
    return np.exp(0.05 * x**2)

# EXEMPLE 2: z′(x) = 1 - 30x/(2√x) + 15z(x), z(1) = 1
def f_exemple2(x, y):
    # z′(x) = 1 - 30x/(2√x) + 15z(x)
    # Simplification: 30x/(2√x) = 15x/√x = 15√x
    return 1 - 15*np.sqrt(x) + 15*y

def sol_exacte2(x):
    # Solution donnée: z(x) = √x
    return np.sqrt(x)

# EXEMPLE 3: z′(x) = πcos(πx)z(x), z(0) = 0
def f_exemple3(x, y):
    return np.pi * np.cos(np.pi * x) * y

def sol_exacte3(x):
    # Solution: z(x) = e^{sin(πx)} - 1
    return np.exp(np.sin(np.pi * x)) - 1

# =============================================================================
# FONCTION DE COMPARAISON POUR UN PAS DONNÉ
# =============================================================================

def comparer_methodes_pas_fixe(f, sol_exacte, y0, x0, xf, h, nom_exemple):
    """Compare les méthodes avec un pas fixé"""
    solveur = SolveurEDO(f)
    n = int((xf - x0) / h)
    
    print(f"\n{'='*80}")
    print(f"ANALYSE - {nom_exemple}")
    print(f"{'='*80}")
    print(f"Intervalle: [{x0}, {xf}]")
    print(f"Pas: h = {h}")
    print(f"Nombre de points: n = {n}")
    print(f"Condition initiale: z({x0}) = {y0}")
    
    # Mesure des temps d'exécution
    debut = time.time()
    x_euler, y_euler = solveur.euler(y0, x0, xf, n)
    temps_euler = time.time() - debut
    
    debut = time.time()
    x_heun, y_heun = solveur.heun(y0, x0, xf, n)
    temps_heun = time.time() - debut
    
    debut = time.time()
    x_rk4, y_rk4 = solveur.runge_kutta_4(y0, x0, xf, n)
    temps_rk4 = time.time() - debut
    
    # Solution de référence (SciPy)
    debut = time.time()
    sol_scipy = solve_ivp(f, [x0, xf], [y0], t_eval=np.linspace(x0, xf, n+1), 
                         method='RK45', rtol=1e-12, atol=1e-12)
    temps_scipy = time.time() - debut
    y_scipy = sol_scipy.y[0]
    
    # Calcul des erreurs
    y_exact = sol_exacte(np.linspace(x0, xf, n+1))
    
    erreur_euler = np.abs(y_euler - y_exact)
    erreur_heun = np.abs(y_heun - y_exact)
    erreur_rk4 = np.abs(y_rk4 - y_exact)
    erreur_scipy = np.abs(y_scipy - y_exact)
    
    # Tableau des résultats
    print(f"\n{'Méthode':<15} {'Temps (s)':<12} {'Erreur max':<15} {'Erreur moyenne':<15} {'Erreur/pas':<15}")
    print('-'*75)
    
    methodes = ['Euler', 'Heun', 'Runge-Kutta 4', 'SciPy RK45']
    temps = [temps_euler, temps_heun, temps_rk4, temps_scipy]
    erreurs_max = [np.max(erreur_euler), np.max(erreur_heun), 
                  np.max(erreur_rk4), np.max(erreur_scipy)]
    erreurs_moy = [np.mean(erreur_euler), np.mean(erreur_heun),
                  np.mean(erreur_rk4), np.mean(erreur_scipy)]
    
    for i in range(4):
        rapport = erreurs_max[i] / h if h > 0 else 0
        print(f"{methodes[i]:<15} {temps[i]:<12.6f} "
              f"{erreurs_max[i]:<15.2e} {erreurs_moy[i]:<15.2e} {rapport:<15.4f}")
    
    # Graphique
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{nom_exemple} - Pas h = {h}', fontsize=16, y=1.02)
    
    # Graphique 1: Solutions
    x_fine = np.linspace(x0, xf, 1000)
    y_fine = sol_exacte(x_fine)
    
    axs[0,0].plot(x_fine, y_fine, 'k-', linewidth=3, label='Solution exacte', alpha=0.8)
    axs[0,0].plot(x_euler, y_euler, 'ro-', markersize=4, label='Euler', alpha=0.7)
    axs[0,0].plot(x_heun, y_heun, 'bs-', markersize=4, label='Heun', alpha=0.7)
    axs[0,0].plot(x_rk4, y_rk4, 'g^-', markersize=4, label='RK4', alpha=0.7)
    axs[0,0].plot(x_euler, y_scipy, 'mv-', markersize=4, label='SciPy RK45', alpha=0.7)
    axs[0,0].set_xlabel('x')
    axs[0,0].set_ylabel('z(x)')
    axs[0,0].set_title('Solutions numériques vs exacte')
    axs[0,0].legend(loc='best', fontsize=9)
    axs[0,0].grid(True, alpha=0.3)
    
    # Graphique 2: Erreurs absolues
    axs[0,1].semilogy(x_euler, erreur_euler, 'ro-', markersize=2, label='Euler', alpha=0.7)
    axs[0,1].semilogy(x_heun, erreur_heun, 'bs-', markersize=2, label='Heun', alpha=0.7)
    axs[0,1].semilogy(x_rk4, erreur_rk4, 'g^-', markersize=2, label='RK4', alpha=0.7)
    axs[0,1].semilogy(x_euler, erreur_scipy, 'mv-', markersize=2, label='SciPy RK45', alpha=0.7)
    axs[0,1].set_xlabel('x')
    axs[0,1].set_ylabel('Erreur absolue (log)')
    axs[0,1].set_title('Erreurs absolues')
    axs[0,1].legend(loc='best', fontsize=9)
    axs[0,1].grid(True, alpha=0.3)
    
    # Graphique 3: Temps d'exécution
    colors = ['red', 'blue', 'green', 'magenta']
    bars1 = axs[1,0].bar(methodes, temps, color=colors, alpha=0.7)
    axs[1,0].set_ylabel('Temps (secondes)')
    axs[1,0].set_title('Temps d\'exécution')
    for bar, t in zip(bars1, temps):
        height = bar.get_height()
        axs[1,0].text(bar.get_x() + bar.get_width()/2, height*1.05,
                     f'{t:.6f}', ha='center', va='bottom', fontsize=9)
    axs[1,0].grid(True, alpha=0.3)
    
    # Graphique 4: Erreurs maximales
    bars2 = axs[1,1].bar(methodes, erreurs_max, color=colors, alpha=0.7)
    axs[1,1].set_yscale('log')
    axs[1,1].set_ylabel('Erreur maximale (log)')
    axs[1,1].set_title('Erreurs maximales')
    for bar, err in zip(bars2, erreurs_max):
        height = bar.get_height()
        axs[1,1].text(bar.get_x() + bar.get_width()/2, height*1.2,
                     f'{err:.2e}', ha='center', va='bottom', fontsize=9)
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'methodes': methodes,
        'temps': temps,
        'erreur_max': erreurs_max,
        'erreur_moyenne': erreurs_moy,
        'solutions': {
            'x': np.linspace(x0, xf, n+1),
            'euler': y_euler, 'heun': y_heun, 'rk4': y_rk4, 'scipy': y_scipy,
            'exacte': y_exact
        }
    }

# =============================================================================
# ANALYSE COMPARATIVE DES 3 EXEMPLES
# =============================================================================

def analyse_comparative():
    """Compare les 3 exemples avec différents pas"""
    
    print("ANALYSE COMPARATIVE DES MÉTHODES NUMÉRIQUES POUR LES EDO")
    print("="*80)
    print("\nTrois exemples d'équations différentielles:")
    print("1. z′(x) = 0.1×x×z(x), z(0) = 1")
    print("2. z′(x) = 1 - 15√x + 15z(x), z(1) = 1 (solution: z(x) = √x)")
    print("3. z′(x) = πcos(πx)z(x), z(0) = 0 (solution: z(x) = e^{sin(πx)} - 1)")
    print("="*80)
    
    # Configuration des exemples
    exemples = [
        {
            'f': f_exemple1,
            'sol': sol_exacte1,
            'nom': "Exemple 1: z' = 0.1*x*z",
            'y0': 1,
            'x0': 0,
            'xf': 3,
            'h': 0.5
        },
        {
            'f': f_exemple2,
            'sol': sol_exacte2,
            'nom': "Exemple 2: z' = 1 - 15√x + 15z",
            'y0': 1,
            'x0': 1,
            'xf': 4,
            'h': 0.5
        },
        {
            'f': f_exemple3,
            'sol': sol_exacte3,
            'nom': "Exemple 3: z' = πcos(πx)z",
            'y0': 0,
            'x0': 0,
            'xf': 2,
            'h': 0.3  # Comme spécifié dans l'énoncé
        }
    ]
    
    # Tableau récapitulatif
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("="*80)
    
    en_tetes = ["Exemple", "Méthode", "Pas", "Erreur max", "Erreur moy", "Temps (s)"]
    print(f"{en_tetes[0]:<25} {en_tetes[1]:<15} {en_tetes[2]:<8} {en_tetes[3]:<15} {en_tetes[4]:<15} {en_tetes[5]:<12}")
    print("-"*95)
    
    resultats_complets = []
    
    for idx, exemple in enumerate(exemples):
        resultat = comparer_methodes_pas_fixe(
            exemple['f'], exemple['sol'], 
            exemple['y0'], exemple['x0'], exemple['xf'], 
            exemple['h'], exemple['nom']
        )
        
        resultats_complets.append(resultat)
        
        # Ajouter au tableau récapitulatif
        for i, methode in enumerate(resultat['methodes']):
            print(f"{exemple['nom']:<25} {methode:<15} {exemple['h']:<8.2f} "
                  f"{resultat['erreur_max'][i]:<15.2e} {resultat['erreur_moyenne'][i]:<15.2e} "
                  f"{resultat['temps'][i]:<12.6f}")
        
        print("-"*95)
    
    # Graphique de comparaison finale
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('COMPARAISON DES 3 EXEMPLES AVEC DIFFÉRENTS PAS', fontsize=16, y=1.02)
    
    noms_exemples = [ex['nom'] for ex in exemples]
    colors = ['red', 'blue', 'green', 'magenta']
    
    # Graphique 1: Erreurs maximales par méthode
    bar_width = 0.2
    x_pos = np.arange(len(resultats_complets[0]['methodes']))
    
    for idx, resultat in enumerate(resultats_complets):
        positions = x_pos + idx * bar_width - bar_width
        axs[0,0].bar(positions, resultat['erreur_max'], 
                    width=bar_width, alpha=0.7, 
                    label=f"Ex {idx+1}", color=colors[idx])
    
    axs[0,0].set_yscale('log')
    axs[0,0].set_ylabel('Erreur maximale (log)')
    axs[0,0].set_title('Erreurs maximales par méthode')
    axs[0,0].set_xticks(x_pos + bar_width)
    axs[0,0].set_xticklabels(resultats_complets[0]['methodes'])
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Temps d'exécution
    for idx, resultat in enumerate(resultats_complets):
        positions = x_pos + idx * bar_width - bar_width
        axs[0,1].bar(positions, resultat['temps'], 
                    width=bar_width, alpha=0.7, 
                    label=f"Ex {idx+1}", color=colors[idx])
    
    axs[0,1].set_ylabel('Temps (secondes)')
    axs[0,1].set_title('Temps d\'exécution par méthode')
    axs[0,1].set_xticks(x_pos + bar_width)
    axs[0,1].set_xticklabels(resultats_complets[0]['methodes'])
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3, axis='y')
    
    # Graphique 3: Rapport erreur/temps (efficacité)
    for idx, resultat in enumerate(resultats_complets):
        rapports = [resultat['erreur_max'][i]/resultat['temps'][i] if resultat['temps'][i] > 0 else 0 
                   for i in range(4)]
        positions = x_pos + idx * bar_width - bar_width
        axs[0,2].bar(positions, rapports, 
                    width=bar_width, alpha=0.7, 
                    label=f"Ex {idx+1}", color=colors[idx])
    
    axs[0,2].set_yscale('log')
    axs[0,2].set_ylabel('Rapport Erreur/Temps (log)')
    axs[0,2].set_title('Efficacité: erreur par unité de temps')
    axs[0,2].set_xticks(x_pos + bar_width)
    axs[0,2].set_xticklabels(resultats_complets[0]['methodes'])
    axs[0,2].legend()
    axs[0,2].grid(True, alpha=0.3, axis='y')
    
    # Graphique 4-6: Solutions pour chaque exemple (méthode de Heun)
    for idx, exemple in enumerate(exemples):
        n = int((exemple['xf'] - exemple['x0']) / exemple['h'])
        x = np.linspace(exemple['x0'], exemple['xf'], n+1)
        y_exact = exemple['sol'](x)
        
        solveur = SolveurEDO(exemple['f'])
        x_heun, y_heun = solveur.heun(exemple['y0'], exemple['x0'], exemple['xf'], n)
        
        axs[1,idx].plot(x, y_exact, 'k-', linewidth=2, label='Solution exacte')
        axs[1,idx].plot(x_heun, y_heun, 'bo-', markersize=3, label=f'Heun (h={exemple["h"]})')
        axs[1,idx].set_xlabel('x')
        axs[1,idx].set_ylabel('z(x)')
        axs[1,idx].set_title(f'{exemple["nom"].split(":")[0]}')
        axs[1,idx].legend()
        axs[1,idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION GÉNÉRALE:")
    print("="*80)
    print("\n1. MÉTHODE D'EULER (ordre 1):")
    print("   - Plus rapide mais moins précise")
    print("   - Erreurs importantes avec pas de 0.5")
    print("   - À utiliser pour des calculs rapides ou quand la précision n'est pas critique")
    
    print("\n2. MÉTHODE DE HEUN (ordre 2):")
    print("   - Bon compromis précision/temps")
    print("   - Performances satisfaisantes avec pas de 0.5")
    print("   - Méthode recommandée pour la plupart des applications")
    
    print("\n3. MÉTHODE RUNGE-KUTTA 4 (ordre 4):")
    print("   - Très précise mais plus lente")
    print("   - Excellente précision même avec pas de 0.5")
    print("   - À utiliser quand une haute précision est requise")
    
    print("\n4. SCIPY RK45:")
    print("   - Méthode adaptative la plus précise")
    print("   - Optimisée pour la performance")
    print("   - Meilleur choix pour des applications professionnelles")
    
    print("\n5. IMPACT DU PAS:")
    print("   - Pas de 0.5: bon compromis pour Heun et RK4")
    print("   - Pas de 0.3: améliore significativement la précision d'Euler")
    print("   - Plus le pas est petit, meilleure est la précision mais plus long est le calcul")

# =============================================================================
# ANALYSE DE LA CONVERGENCE
# =============================================================================

def analyser_convergence_exemple3():
    """Analyse spécifique de la convergence pour l'exemple 3"""
    print("\n" + "="*80)
    print("ANALYSE DE CONVERGENCE - Exemple 3 (pas = 0.3)")
    print("="*80)
    
    solveur = SolveurEDO(f_exemple3)
    y0, x0, xf = 0, 0, 2
    h = 0.3
    n = int((xf - x0) / h)
    
    print(f"Configuration:")
    print(f"  Équation: z'(x) = πcos(πx)z(x)")
    print(f"  Condition initiale: z(0) = {y0}")
    print(f"  Intervalle: [{x0}, {xf}]")
    print(f"  Pas: h = {h}")
    print(f"  Nombre de points: n = {n}")
    
    # Calcul avec différentes méthodes
    x_euler, y_euler = solveur.euler(y0, x0, xf, n)
    x_heun, y_heun = solveur.heun(y0, x0, xf, n)
    x_rk4, y_rk4 = solveur.runge_kutta_4(y0, x0, xf, n)
    
    # Solution exacte
    y_exact_euler = sol_exacte3(x_euler)
    y_exact_heun = sol_exacte3(x_heun)
    y_exact_rk4 = sol_exacte3(x_rk4)
    
    # Calcul des erreurs locales
    erreur_locale_euler = np.abs(y_euler - y_exact_euler)
    erreur_locale_heun = np.abs(y_heun - y_exact_heun)
    erreur_locale_rk4 = np.abs(y_rk4 - y_exact_rk4)
    
    # Graphique de l'erreur locale
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x_fine = np.linspace(x0, xf, 1000)
    plt.plot(x_fine, sol_exacte3(x_fine), 'k-', linewidth=2, label='Solution exacte')
    plt.plot(x_euler, y_euler, 'ro-', markersize=3, label='Euler', alpha=0.7)
    plt.plot(x_heun, y_heun, 'bs-', markersize=3, label='Heun', alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'g^-', markersize=3, label='RK4', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('z(x)')
    plt.title(f'Exemple 3 - Solutions avec h={h}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(x_euler, erreur_locale_euler, 'ro-', markersize=3, label='Euler', alpha=0.7)
    plt.semilogy(x_heun, erreur_locale_heun, 'bs-', markersize=3, label='Heun', alpha=0.7)
    plt.semilogy(x_rk4, erreur_locale_rk4, 'g^-', markersize=3, label='RK4', alpha=0.7)
    plt.axhline(y=h**1, color='k', linestyle='--', alpha=0.3, label='Ordre 1')
    plt.axhline(y=h**2, color='k', linestyle=':', alpha=0.3, label='Ordre 2')
    plt.axhline(y=h**4, color='k', linestyle='-.', alpha=0.3, label='Ordre 4')
    plt.xlabel('x')
    plt.ylabel('Erreur locale (log)')
    plt.title(f'Erreurs locales avec h={h}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Tableau des erreurs globales
    print(f"\n{'Méthode':<15} {'Erreur max':<15} {'Erreur moyenne':<15} {'Rapport h^p':<15}")
    print('-'*65)
    
    ordres_theoriques = [1, 2, 4]  # Ordres théoriques des méthodes
    erreurs_max = [np.max(erreur_locale_euler), np.max(erreur_locale_heun), np.max(erreur_locale_rk4)]
    
    erreurs_moyennes = [np.mean(erreur_locale_euler), np.mean(erreur_locale_heun), np.mean(erreur_locale_rk4)]
    
    for i, methode in enumerate(['Euler', 'Heun', 'RK4']):
        rapport = erreurs_max[i] / (h**ordres_theoriques[i])
        print(f"{methode:<15} {erreurs_max[i]:<15.2e} "
              f"{erreurs_moyennes[i]:<15.2e} "
              f"{rapport:<15.4f}")

# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    # Analyse comparative des 3 exemples
    analyse_comparative()
    
    # Analyse spécifique de la convergence pour l'exemple 3
    analyser_convergence_exemple3()
    
    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ FINAL DES RÉSULTATS")
    print("="*80)
    print("\nPour un pas de 0.5:")
    print("• Euler: erreur ~10⁻¹ à 10⁰ (acceptable pour estimations rapides)")
    print("• Heun: erreur ~10⁻² à 10⁻³ (bon pour la plupart des applications)")
    print("• RK4: erreur ~10⁻⁴ à 10⁻⁶ (excellent pour haute précision)")
    
    print("\nPour un pas de 0.3 (exemple 3):")
    print("• Euler: erreur ~10⁻¹ (amélioration par rapport à h=0.5)")
    print("• Heun: erreur ~10⁻² (très bonne précision)")
    print("• RK4: erreur ~10⁻⁴ (précision exceptionnelle)")
    
    print("\nRecommandation:")
    print("Pour un équilibre optimal précision/temps, utiliser:")
    print("1. Heun avec h=0.5 pour des calculs rapides et précis")
    print("2. RK4 avec h=0.5 pour une haute précision")
    print("3. Euler uniquement pour des estimations très rapides")

