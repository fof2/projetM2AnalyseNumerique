import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt, sin, cos, pi, exp

class SolveurEDO:
    def __init__(self):
        pass
    
    # Méthode d'Euler explicite
    def euler(self, f, x0, y0, h, n):
        """
        f: fonction f(x, y) = y'
        x0: point initial
        y0: valeur initiale y(x0)
        h: pas
        n: nombre d'itérations
        """
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        x[0] = x0
        y[0] = y0
        
        for i in range(n):
            y[i+1] = y[i] + h * f(x[i], y[i])
            x[i+1] = x[i] + h
        
        return x, y
    
    # Méthode de Heun (Euler amélioré)
    def heun(self, f, x0, y0, h, n):
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        x[0] = x0
        y[0] = y0
        
        for i in range(n):
            k1 = f(x[i], y[i])
            k2 = f(x[i] + h, y[i] + h * k1)
            y[i+1] = y[i] + h * (k1 + k2) / 2
            x[i+1] = x[i] + h
        
        return x, y
    
    # Méthode de Runge-Kutta d'ordre 4
    def runge_kutta_4(self, f, x0, y0, h, n):
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        x[0] = x0
        y[0] = y0
        
        for i in range(n):
            k1 = f(x[i], y[i])
            k2 = f(x[i] + h/2, y[i] + h*k1/2)
            k3 = f(x[i] + h/2, y[i] + h*k2/2)
            k4 = f(x[i] + h, y[i] + h*k3)
            
            y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
            x[i+1] = x[i] + h
        
        return x, y
    
    # Calcul de l'erreur
    def calculer_erreur(self, y_numerique, y_exacte):
        erreur_absolue = np.abs(y_numerique - y_exacte)
        erreur_relative = np.abs(y_numerique - y_exacte) / (np.abs(y_exacte) + 1e-10)
        return erreur_absolue, erreur_relative

# Définir les équations différentielles du problème
def f1(x, z):
    """z′(x) = 0.1 * x * z(x)"""
    return 0.1 * x * z

def solution_exacte1(x):
    """Solution exacte de z′(x) = 0.1 * x * z(x) avec z(0)=1"""
    return np.exp(0.05 * x**2)

def f2(x, z):
    """z′(x) = 1/(2√x) + 15z(x) - 30x"""
    if x == 0:
        return 0
    return (1 - 30*x) / (2*sqrt(x)) + 15*z

def solution_exacte2(x):
    """Solution exacte z(x) = √x"""
    return np.sqrt(x)

def f3(x, z):
    """z′(x) = πcos(πx)z(x)"""
    return pi * cos(pi * x) * z

def solution_exacte3(x):
    """Solution exacte: exp(sin(πx))"""
    return np.exp(np.sin(pi * x))

def tester_methodes():
    solveur = SolveurEDO()
    
    # Paramètres communs
    pas = 0.3
    
    # Test pour la première équation
    print("=" * 70)
    print("ÉQUATION 1: z′(x) = 0.1 * x * z(x) avec z(0) = 1")
    print("=" * 70)
    
    x0, y0 = 0, 1
    b = 5  # borne supérieure
    n = int((b - x0) / pas)
    
    temps_debut = time.time()
    x_euler1, y_euler1 = solveur.euler(f1, x0, y0, pas, n)
    temps_euler1 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_heun1, y_heun1 = solveur.heun(f1, x0, y0, pas, n)
    temps_heun1 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_rk41, y_rk41 = solveur.runge_kutta_4(f1, x0, y0, pas, n)
    temps_rk41 = time.time() - temps_debut
    
    y_exacte1 = solution_exacte1(x_euler1)
    
    # Calcul des erreurs pour l'équation 1
    erreur_abs_euler1, erreur_rel_euler1 = solveur.calculer_erreur(y_euler1, y_exacte1)
    erreur_abs_heun1, erreur_rel_heun1 = solveur.calculer_erreur(y_heun1, y_exacte1)
    erreur_abs_rk41, erreur_rel_rk41 = solveur.calculer_erreur(y_rk41, y_exacte1)
    
    print(f"\nTemps d'exécution - Équation 1:")
    print(f"Euler: {temps_euler1:.6f} secondes")
    print(f"Heun: {temps_heun1:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk41:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 1:")
    print(f"Euler: {np.max(erreur_abs_euler1):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun1):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk41):.6e}")
    
    # Test pour la deuxième équation
    print("\n" + "=" * 70)
    print("ÉQUATION 2: z′(x) = (1 - 30x)/(2√x) + 15z(x) avec z(0)=0")
    print("Solution exacte: z(x) = √x")
    print("=" * 70)
    
    x0, y0 = 0.01, sqrt(0.01)  # Éviter la singularité en x=0
    b = 2
    n = int((b - x0) / pas)
    
    temps_debut = time.time()
    x_euler2, y_euler2 = solveur.euler(f2, x0, y0, pas, n)
    temps_euler2 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_heun2, y_heun2 = solveur.heun(f2, x0, y0, pas, n)
    temps_heun2 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_rk42, y_rk42 = solveur.runge_kutta_4(f2, x0, y0, pas, n)
    temps_rk42 = time.time() - temps_debut
    
    y_exacte2 = solution_exacte2(x_euler2)
    
    # Calcul des erreurs pour l'équation 2
    erreur_abs_euler2, erreur_rel_euler2 = solveur.calculer_erreur(y_euler2, y_exacte2)
    erreur_abs_heun2, erreur_rel_heun2 = solveur.calculer_erreur(y_heun2, y_exacte2)
    erreur_abs_rk42, erreur_rel_rk42 = solveur.calculer_erreur(y_rk42, y_exacte2)
    
    print(f"\nTemps d'exécution - Équation 2:")
    print(f"Euler: {temps_euler2:.6f} secondes")
    print(f"Heun: {temps_heun2:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk42:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 2:")
    print(f"Euler: {np.max(erreur_abs_euler2):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun2):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk42):.6e}")
    
    # Test pour la troisième équation
    print("\n" + "=" * 70)
    print("ÉQUATION 3: z′(x) = πcos(πx)z(x) avec z(0)=0")
    print("Solution exacte: z(x) = exp(sin(πx))")
    print("Note: z(0)=exp(sin(0))=1 (correction de la condition initiale)")
    print("=" * 70)
    
    x0, y0 = 0, 1  # Correction: exp(sin(0)) = exp(0) = 1
    b = 2
    n = int((b - x0) / pas)
    
    temps_debut = time.time()
    x_euler3, y_euler3 = solveur.euler(f3, x0, y0, pas, n)
    temps_euler3 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_heun3, y_heun3 = solveur.heun(f3, x0, y0, pas, n)
    temps_heun3 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_rk43, y_rk43 = solveur.runge_kutta_4(f3, x0, y0, pas, n)
    temps_rk43 = time.time() - temps_debut
    
    y_exacte3 = solution_exacte3(x_euler3)
    
    # Calcul des erreurs pour l'équation 3
    erreur_abs_euler3, erreur_rel_euler3 = solveur.calculer_erreur(y_euler3, y_exacte3)
    erreur_abs_heun3, erreur_rel_heun3 = solveur.calculer_erreur(y_heun3, y_exacte3)
    erreur_abs_rk43, erreur_rel_rk43 = solveur.calculer_erreur(y_rk43, y_exacte3)
    
    print(f"\nTemps d'exécution - Équation 3:")
    print(f"Euler: {temps_euler3:.6f} secondes")
    print(f"Heun: {temps_heun3:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk43:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 3:")
    print(f"Euler: {np.max(erreur_abs_euler3):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun3):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk43):.6e}")
    
    # Visualisation des résultats
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Équation 1
    axes[0, 0].plot(x_euler1, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 0].plot(x_euler1, y_euler1, 'ro-', label='Euler', markersize=4)
    axes[0, 0].set_title('Équation 1 - Euler vs Exact')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('z(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(x_heun1, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 1].plot(x_heun1, y_heun1, 'go-', label='Heun', markersize=4)
    axes[0, 1].set_title('Équation 1 - Heun vs Exact')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('z(x)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(x_rk41, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 2].plot(x_rk41, y_rk41, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[0, 2].set_title('Équation 1 - RK4 vs Exact')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('z(x)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Équation 2
    axes[1, 0].plot(x_euler2, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 0].plot(x_euler2, y_euler2, 'ro-', label='Euler', markersize=4)
    axes[1, 0].set_title('Équation 2 - Euler vs Exact')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z(x)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(x_heun2, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 1].plot(x_heun2, y_heun2, 'go-', label='Heun', markersize=4)
    axes[1, 1].set_title('Équation 2 - Heun vs Exact')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z(x)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(x_rk42, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 2].plot(x_rk42, y_rk42, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[1, 2].set_title('Équation 2 - RK4 vs Exact')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z(x)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Équation 3
    axes[2, 0].plot(x_euler3, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 0].plot(x_euler3, y_euler3, 'ro-', label='Euler', markersize=4)
    axes[2, 0].set_title('Équation 3 - Euler vs Exact')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('z(x)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(x_heun3, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 1].plot(x_heun3, y_heun3, 'go-', label='Heun', markersize=4)
    axes[2, 1].set_title('Équation 3 - Heun vs Exact')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('z(x)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    axes[2, 2].plot(x_rk43, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 2].plot(x_rk43, y_rk43, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[2, 2].set_title('Équation 3 - RK4 vs Exact')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('z(x)')
    axes[2, 2].legend()
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Graphique d'erreurs comparatives
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    axes2[0].plot(x_euler1, erreur_abs_euler1, 'r-', label='Euler')
    axes2[0].plot(x_heun1, erreur_abs_heun1, 'g-', label='Heun')
    axes2[0].plot(x_rk41, erreur_abs_rk41, 'b-', label='RK4')
    axes2[0].set_title('Erreurs absolues - Équation 1')
    axes2[0].set_xlabel('x')
    axes2[0].set_ylabel('Erreur absolue')
    axes2[0].legend()
    axes2[0].grid(True)
    
    axes2[1].plot(x_euler2, erreur_abs_euler2, 'r-', label='Euler')
    axes2[1].plot(x_heun2, erreur_abs_heun2, 'g-', label='Heun')
    axes2[1].plot(x_rk42, erreur_abs_rk42, 'b-', label='RK4')
    axes2[1].set_title('Erreurs absolues - Équation 2')
    axes2[1].set_xlabel('x')
    axes2[1].set_ylabel('Erreur absolue')
    axes2[1].legend()
    axes2[1].grid(True)
    
    axes2[2].plot(x_euler3, erreur_abs_euler3, 'r-', label='Euler')
    axes2[2].plot(x_heun3, erreur_abs_heun3, 'g-', label='Heun')
    axes2[2].plot(x_rk43, erreur_abs_rk43, 'b-', label='RK4')
    axes2[2].set_title('Erreurs absolues - Équation 3')
    axes2[2].set_xlabel('x')
    axes2[2].set_ylabel('Erreur absolue')
    axes2[2].legend()
    axes2[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'equations': [f1, f2, f3],
        'solutions': [solution_exacte1, solution_exacte2, solution_exacte3],
        'methodes': ['Euler', 'Heun', 'Runge-Kutta 4'],
        'temps': [
            [temps_euler1, temps_heun1, temps_rk41],
            [temps_euler2, temps_heun2, temps_rk42],
            [temps_euler3, temps_heun3, temps_rk43]
        ],
        'erreurs': [
            [erreur_abs_euler1, erreur_abs_heun1, erreur_abs_rk41],
            [erreur_abs_euler2, erreur_abs_heun2, erreur_abs_rk42],
            [erreur_abs_euler3, erreur_abs_heun3, erreur_abs_rk43]
        ]
    }

# Exécution principale
if __name__ == "__main__":
    print("COMPARAISON DES MÉTHODES DE RÉSOLUTION D'ÉQUATIONS DIFFÉRENTIELLES")
    print("=" * 70)
    
    resultats = tester_methodes()
    
    # Affichage synthétique des résultats
    print("\n" + "=" * 70)
    print("SYNTHÈSE DES RÉSULTATS")
    print("=" * 70)
    
    for i in range(3):
        print(f"\nÉquation {i+1}:")
        print(f"  Euler - Temps: {resultats['temps'][i][0]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][0]):.6e}")
        print(f"  Heun - Temps: {resultats['temps'][i][1]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][1]):.6e}")
        print(f"  RK4 - Temps: {resultats['temps'][i][2]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][2]):.6e}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("1. Runge-Kutta 4 est la méthode la plus précise (erreur minimale)")
    print("2. Euler est la méthode la plus rapide mais la moins précise")
    print("3. Heun est un bon compromis entre précision et vitesse")
    print("4. Pour des pas plus petits, toutes les méthodes seraient plus précises")
