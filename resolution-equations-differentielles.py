resolution-equations-differentielles

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
            y_pred = y[i] + h * self.f(x[i], y[i])
            y[i+1] = y[i] + h/2 * (self.f(x[i], y[i]) + self.f(x[i+1], y_pred))
        
        return x, y
    
    def runge_kutta_4(self, y0, x0, xf, n):
        h = (xf - x0) / n
        x = np.linspace(x0, xf, n+1)
        y = np.zeros(n+1)
        y[0] = y0
        
        for i in range(n):
            k1 = h * self.f(x[i], y[i])
            k2 = h * self.f(x[i] + h/2, y[i] + k1/2)
            k3 = h * self.f(x[i] + h/2, y[i] + k2/2)
            k4 = h * self.f(x[i] + h, y[i] + k3)
            
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return x, y

def comparer_methodes(f, sol_exacte, y0, x0, xf, n, nom_exemple):
    solveur = SolveurEDO(f)
    
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
    
    # Solution de référence (méthode de SciPy)
    debut = time.time()
    sol_scipy = solve_ivp(f, [x0, xf], [y0], t_eval=np.linspace(x0, xf, n+1), method='RK45', rtol=1e-8)
    temps_scipy = time.time() - debut
    y_scipy = sol_scipy.y[0]
    
    # Calcul des erreurs
    y_exact_euler = sol_exacte(x_euler)
    y_exact_heun = sol_exacte(x_heun)
    y_exact_rk4 = sol_exacte(x_rk4)
    y_exact_scipy = sol_exacte(sol_scipy.t)
    
    erreur_euler = np.abs(y_euler - y_exact_euler)
    erreur_heun = np.abs(y_heun - y_exact_heun)
    erreur_rk4 = np.abs(y_rk4 - y_exact_rk4)
    erreur_scipy = np.abs(y_scipy - y_exact_scipy)
    
    return {
        'methodes': ['Euler', 'Heun', 'Runge-Kutta 4', 'SciPy RK45'],
        'temps': [temps_euler, temps_heun, temps_rk4, temps_scipy],
        'erreur_max': [
            np.max(erreur_euler),
            np.max(erreur_heun),
            np.max(erreur_rk4),
            np.max(erreur_scipy)
        ],
        'erreur_moyenne': [
            np.mean(erreur_euler),
            np.mean(erreur_heun),
            np.mean(erreur_rk4),
            np.mean(erreur_scipy)
        ],
        'solutions': {
            'x_euler': x_euler, 'x_heun': x_heun, 'x_rk4': x_rk4, 'x_scipy': sol_scipy.t,
            'euler': y_euler, 'heun': y_heun, 'rk4': y_rk4, 'scipy': y_scipy,
            'exacte_euler': y_exact_euler, 'exacte_heun': y_exact_heun,
            'exacte_rk4': y_exact_rk4, 'exacte_scipy': y_exact_scipy
        }
    }

def afficher_comparaison(resultats, nom_exemple):
    print(f"\n{'='*80}")
    print(f"COMPARAISON COMPLÈTE - {nom_exemple}")
    print(f"{'='*80}")
    
    # Tableau des performances
    print(f"\n{'Méthode':<15} {'Temps (s)':<12} {'Erreur max':<15} {'Erreur moyenne':<15}")
    print('-'*65)
    for i in range(len(resultats['methodes'])):
        print(f"{resultats['methodes'][i]:<15} {resultats['temps'][i]:<12.6f} "
              f"{resultats['erreur_max'][i]:<15.2e} {resultats['erreur_moyenne'][i]:<15.2e}")
    
    # Graphique des solutions
    plt.figure(figsize=(15, 10))
    
    # Solutions
    plt.subplot(2, 2, 1)
    x_fin = np.linspace(resultats['solutions']['x_euler'][0], resultats['solutions']['x_euler'][-1], 1000)
    y_exact_fin = sol_exacte1(x_fin)
    plt.plot(x_fin, y_exact_fin, 'k-', linewidth=2, label='Solution exacte')
    plt.plot(resultats['solutions']['x_euler'], resultats['solutions']['euler'], 'ro-', 
             markersize=3, label='Euler', alpha=0.7, linewidth=1)
    plt.plot(resultats['solutions']['x_heun'], resultats['solutions']['heun'], 'bs-', 
             markersize=3, label='Heun', alpha=0.7, linewidth=1)
    plt.plot(resultats['solutions']['x_rk4'], resultats['solutions']['rk4'], 'g^-', 
             markersize=3, label='Runge-Kutta 4', alpha=0.7, linewidth=1)
    plt.plot(resultats['solutions']['x_scipy'], resultats['solutions']['scipy'], 'mv-', 
             markersize=3, label='SciPy RK45', alpha=0.7, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'Solutions - {nom_exemple}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Erreurs
    plt.subplot(2, 2, 2)
    plt.semilogy(resultats['solutions']['x_euler'], 
                 np.abs(resultats['solutions']['euler'] - resultats['solutions']['exacte_euler']), 
                 'ro-', label='Euler', alpha=0.7, markersize=2, linewidth=1)
    plt.semilogy(resultats['solutions']['x_heun'], 
                 np.abs(resultats['solutions']['heun'] - resultats['solutions']['exacte_heun']), 
                 'bs-', label='Heun', alpha=0.7, markersize=2, linewidth=1)
    plt.semilogy(resultats['solutions']['x_rk4'], 
                 np.abs(resultats['solutions']['rk4'] - resultats['solutions']['exacte_rk4']), 
                 'g^-', label='Runge-Kutta 4', alpha=0.7, markersize=2, linewidth=1)
    plt.semilogy(resultats['solutions']['x_scipy'], 
                 np.abs(resultats['solutions']['scipy'] - resultats['solutions']['exacte_scipy']), 
                 'mv-', label='SciPy RK45', alpha=0.7, markersize=2, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('Erreur absolue (échelle log)')
    plt.title('Erreurs absolues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Temps d'exécution
    plt.subplot(2, 2, 3)
    methodes = resultats['methodes']
    temps = resultats['temps']
    colors = ['red', 'blue', 'green', 'magenta']
    bars = plt.bar(methodes, temps, color=colors, alpha=0.7)
    plt.ylabel('Temps (secondes)')
    plt.title('Temps d\'exécution')
    for bar, t in zip(bars, temps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.6f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Erreurs maximales
    plt.subplot(2, 2, 4)
    erreurs = resultats['erreur_max']
    bars = plt.bar(methodes, erreurs, color=colors, alpha=0.7)
    plt.yscale('log')
    plt.ylabel('Erreur maximale (échelle log)')
    plt.title('Erreurs maximales')
    for bar, err in zip(bars, erreurs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.2e}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FONCTIONS TEST AVEC MEILLEURS PARAMÈTRES
# =============================================================================
def f1(x, y):
    """y' = -2x*y - Équation plus stable"""
    return -2 * x * y

def sol_exacte1(x):
    return np.exp(-x**2)

def f2(x, y):
    """Équation plus simple pour mieux voir la convergence"""
    return -y  # Solution: y(x) = exp(-x)

def sol_exacte2(x):
    return np.exp(-x)

def f3(x, y):
    """Équation linéaire simple"""
    return x - y  # Solution: y(x) = (x-1) + 2*exp(-x) pour y(0)=1

def sol_exacte3(x):
    return (x - 1) + 2*np.exp(-x)

# =============================================================================
# DÉMONSTRATION AVEC DIFFÉRENTS PAS
# =============================================================================
def demonstration_pas_variables():
    """Montre l'effet du nombre de pas sur la précision"""
    print("\n" + "="*80)
    print("DÉMONSTRATION: EFFET DU NOMBRE DE PAS SUR LA PRÉCISION")
    print("="*80)
    
    f = f1
    sol_exacte = sol_exacte1
    y0, x0, xf = 1, 0, 2
    
    pas_liste = [10, 20, 50, 100, 200]
    
    plt.figure(figsize=(15, 10))
    
    for i, n in enumerate(pas_liste):
        solveur = SolveurEDO(f)
        
        x_euler, y_euler = solveur.euler(y0, x0, xf, n)
        x_heun, y_heun = solveur.heun(y0, x0, xf, n)
        x_rk4, y_rk4 = solveur.runge_kutta_4(y0, x0, xf, n)
        
        # Solution exacte fine
        x_exact = np.linspace(x0, xf, 1000)
        y_exact = sol_exacte(x_exact)
        
        plt.subplot(2, 3, i+1)
        plt.plot(x_exact, y_exact, 'k-', linewidth=2, label='Exacte')
        plt.plot(x_euler, y_euler, 'ro-', markersize=2, label=f'Euler (n={n})', alpha=0.7)
        plt.plot(x_heun, y_heun, 'bs-', markersize=2, label=f'Heun (n={n})', alpha=0.7)
        plt.plot(x_rk4, y_rk4, 'g^-', markersize=2, label=f'RK4 (n={n})', alpha=0.7)
        
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.title(f'{n} pas - h={(xf-x0)/n:.3f}')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Tableau d'erreurs
    print(f"\n{'n':<6} {'h':<8} {'Euler':<12} {'Heun':<12} {'RK4':<12}")
    print('-'*50)
    for n in pas_liste:
        h = (xf - x0) / n
        solveur = SolveurEDO(f)
        
        x_e, y_e = solveur.euler(y0, x0, xf, n)
        x_h, y_h = solveur.heun(y0, x0, xf, n)
        x_r, y_r = solveur.runge_kutta_4(y0, x0, xf, n)
        
        err_e = np.max(np.abs(y_e - sol_exacte(x_e)))
        err_h = np.max(np.abs(y_h - sol_exacte(x_h)))
        err_r = np.max(np.abs(y_r - sol_exacte(x_r)))
        
        print(f"{n:<6} {h:<8.3f} {err_e:<12.2e} {err_h:<12.2e} {err_r:<12.2e}")

# =============================================================================
# EXÉCUTION PRINCIPALE AVEC BONS PARAMÈTRES
# =============================================================================
if __name__ == "__main__":
    print("COMPARAISON DES MÉTHODES - SOLUTIONS CONVERGENTES")
    print("=" * 80)
    
    # Exemple 1 avec plus de pas pour mieux voir la convergence
    print("\n>>> Exemple 1: y' = -2x*y, y(0) = 1")
    print("    Intervalle [0, 1] au lieu de [0, 2] pour plus de stabilité")
    resultats1 = comparer_methodes(f1, sol_exacte1, 1, 0, 1, 50, "y' = -2x*y sur [0,1]")
    afficher_comparaison(resultats1, "y' = -2x*y sur [0,1]")
    
    # Exemple 2 - Équation plus simple
    print("\n>>> Exemple 2: y' = -y, y(0) = 1")
    resultats2 = comparer_methodes(f2, sol_exacte2, 1, 0, 3, 60, "y' = -y")
    afficher_comparaison(resultats2, "y' = -y")
    
    # Exemple 3 - Équation linéaire
    print("\n>>> Exemple 3: y' = x - y, y(0) = 1")
    resultats3 = comparer_methodes(f3, sol_exacte3, 1, 0, 3, 60, "y' = x - y")
    afficher_comparaison(resultats3, "y' = x - y")
    
    # Démonstration de l'effet du pas
    demonstration_pas_variables()
    
    # Test avec beaucoup de pas pour montrer la convergence
    print("\n" + "="*80)
    print("TEST AVEC NOMBRE DE PAS ÉLEVÉ (n=200)")
    print("="*80)
    
    solveur = SolveurEDO(f1)
    y0, x0, xf = 1, 0, 2
    n = 200
    
    x_e, y_e = solveur.euler(y0, x0, xf, n)
    x_h, y_h = solveur.heun(y0, x0, xf, n)
    x_r, y_r = solveur.runge_kutta_4(y0, x0, xf, n)
    x_exact = np.linspace(x0, xf, 1000)
    y_exact = sol_exacte1(x_exact)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_exact, y_exact, 'k-', linewidth=3, label='Solution exacte')
    plt.plot(x_e, y_e, 'ro-', markersize=2, label='Euler (n=200)', alpha=0.7)
    plt.plot(x_h, y_h, 'bs-', markersize=2, label='Heun (n=200)', alpha=0.7)
    plt.plot(x_r, y_r, 'g^-', markersize=2, label='RK4 (n=200)', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Convergence avec n=200 pas - Méthodes proches de la solution exacte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    err_e = np.max(np.abs(y_e - sol_exacte1(x_e)))
    err_h = np.max(np.abs(y_h - sol_exacte1(x_h)))
    err_r = np.max(np.abs(y_r - sol_exacte1(x_r)))
    
    print(f"Erreurs maximales avec n=200:")
    print(f"Euler:  {err_e:.2e}")
    print(f"Heun:   {err_h:.2e}")
    print(f"RK4:    {err_r:.2e}")
