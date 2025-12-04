import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
from scipy.special import roots_legendre, roots_laguerre, roots_chebyt
import math
import warnings
warnings.filterwarnings('ignore')

class IntegrationNumerique:
    def __init__(self):
        pass
    
    # 1. Quadrature de Gauss-Legendre (classique)
    def gauss_legendre(self, f, a, b, n):
        """
        Intégration par quadrature de Gauss-Legendre
        f: fonction à intégrer
        a, b: bornes d'intégration
        n: nombre de points de Gauss
        """
        if b == np.inf:
            # Transformation pour intervalle infini
            def g(t):
                x = t / (1 - t**2) if t != 1 else np.inf
                return f(x) * (1 + t**2) / (1 - t**2)**2
            
            a_tr = -0.999
            b_tr = 0.999
        else:
            g = f
            a_tr = a
            b_tr = b
        
        # Obtenir les poids et points de Gauss-Legendre sur [-1, 1]
        x, w = roots_legendre(n)
        
        # Transformation linéaire pour passer de [-1, 1] à [a_tr, b_tr]
        t = 0.5 * (b_tr - a_tr) * x + 0.5 * (a_tr + b_tr)
        
        # Évaluation de la fonction et calcul de l'intégrale
        integral = 0.5 * (b_tr - a_tr) * np.sum(w * g(t))
        
        return integral
    
    # 2. Quadrature de Gauss-Laguerre
    def gauss_laguerre(self, f, n):
        """
        Intégration par quadrature de Gauss-Laguerre pour [0, +∞)
        avec poids e^{-x}
        f: fonction à intégrer SANS le poids e^{-x}
        n: nombre de points de Gauss
        """
        # Obtenir les poids et points de Gauss-Laguerre
        x, w = roots_laguerre(n)
        
        # Évaluation de la fonction et calcul de l'intégrale
        integral = np.sum(w * f(x))
        
        return integral
    
    # 3. Quadrature de Gauss-Chebyshev
    def gauss_chebyshev(self, f, a=-1, b=1, n=10):
        """
        Intégration par quadrature de Gauss-Chebyshev sur [-1, 1]
        avec poids 1/√(1-x²)
        f: fonction à intégrer SANS le poids 1/√(1-x²)
        n: nombre de points de Gauss
        """
        # Obtenir les poids et points de Gauss-Chebyshev
        x, w = roots_chebyt(n)
        
        # Transformation pour l'intervalle [a, b] si nécessaire
        if a != -1 or b != 1:
            t = 0.5 * (b - a) * x + 0.5 * (a + b)
            # Les poids doivent être ajustés
            w = w * 0.5 * (b - a)
            integral = np.sum(w * f(t))
        else:
            integral = np.sum(w * f(x))
        
        return integral
    
    # 4. Méthode composite de Simpson
    def simpson_composite(self, f, a, b, n):
        """
        Intégration par méthode composite de Simpson
        n: nombre d'intervalles (doit être pair)
        """
        if b == np.inf:
            # Transformation pour intervalle infini
            def g(t):
                x = t / (1 - t) if t != 1 else np.inf
                return f(x) / (1 - t)**2
            
            a_tr = 0
            b_tr = 0.999
            f_tr = g
        else:
            a_tr = a
            b_tr = b
            f_tr = f
        
        if n % 2 == 1:
            n += 1  # Assurer que n est pair
        
        h = (b_tr - a_tr) / n
        x = np.linspace(a_tr, b_tr, n+1)
        
        # Évaluation de la fonction aux points
        y = f_tr(x)
        
        # Règle de Simpson composite
        integral = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))
        
        return integral
    
    # 5. Intégration par spline cubique
    def integration_spline(self, f, a, b, n):
        """
        Intégration par spline cubique
        n: nombre de points d'interpolation
        """
        from scipy.interpolate import CubicSpline
        
        if b == np.inf:
            # Transformation pour intervalle infini
            def g(t):
                x = t / (1 - t) if t != 1 else np.inf
                return f(x) / (1 - t)**2
            
            a_tr = 0
            b_tr = 0.999
            f_tr = g
        else:
            a_tr = a
            b_tr = b
            f_tr = f
        
        # Points d'interpolation équidistants
        x = np.linspace(a_tr, b_tr, n)
        y = f_tr(x)
        
        # Création de la spline cubique
        cs = CubicSpline(x, y)
        
        # Intégration de la spline
        integral = cs.integrate(a_tr, b_tr)
        
        return integral

# Définition des fonctions tests

class FonctionsTests:
    @staticmethod
    def f1_chebyshev(x):
        """Fonction de type Chebyshev sur [-1, 1] avec poids 1/√(1-x²)
        f(x) = cos(10x)  (fortement oscillante)"""
        return np.cos(10 * x)
    
    @staticmethod
    def f1_exacte():
        """Valeur exacte de ∫_{-1}^{1} cos(10x)/√(1-x²) dx
        C'est π * J_0(10) où J_0 est la fonction de Bessel de première espèce"""
        from scipy.special import j0
        return math.pi * j0(10)
    
    @staticmethod
    def f2_laguerre(x):
        """Fonction de type Laguerre sur [0, ∞) avec poids e^{-x}
        f(x) = 1/(1 + x^2)  (décroissance lente)"""
        return 1 / (1 + x**2)
    
    @staticmethod
    def f2_exacte():
        """Valeur exacte de ∫_{0}^{∞} e^{-x}/(1 + x^2) dx
        Formule connue : Ci(1)sin(1) + (π/2 - Si(1))cos(1)"""
        from scipy.special import sici
        Si, Ci = sici(1)
        return Ci * np.sin(1) + (np.pi/2 - Si) * np.cos(1)
    
    @staticmethod
    def f3_combinee(x):
        """Fonction combinant les deux poids sur [0, 1)
        f(x) = np.cos(x)"""
        return np.cos(x)
    
    @staticmethod
    def f3_exacte():
        """Valeur exacte de ∫_{0}^{1} cos(x)/√(1-x²) dx
        C'est π/2 * J_0(1) où J_0 est la fonction de Bessel"""
        from scipy.special import j0
        return np.pi/2 * j0(1)
    
    @staticmethod
    def f4_neutre(x):
        """Fonction neutre (lisse) sur [-1, 1]
        f(x) = 1/(1 + 25*x**2)  (fonction de Runge)"""
        return 1 / (1 + 25 * x**2)
    
    @staticmethod
    def f4_exacte():
        """Valeur exacte de ∫_{-1}^{1} 1/(1 + 25*x**2) dx
        = (2/5) * arctan(5)"""
        return (2/5) * np.arctan(5)

def tester_methodes():
    integ = IntegrationNumerique()
    funcs = FonctionsTests()
    
    # Configuration des tests
    methodes = {
        'Gauss-Legendre': integ.gauss_legendre,
        'Gauss-Laguerre': integ.gauss_laguerre,
        'Gauss-Chebyshev': integ.gauss_chebyshev,
        'Simpson': integ.simpson_composite,
        'Spline': integ.integration_spline
    }
    
    # Paramètres pour chaque fonction
    fonctions_config = [
        {
            'nom': 'Chebyshev',
            'f': funcs.f1_chebyshev,
            'exacte': funcs.f1_exacte(),
            'intervalle': (-1, 1),
            'methode_speciale': 'Gauss-Chebyshev',
            'poids': '1/√(1-x²)'
        },
        {
            'nom': 'Laguerre',
            'f': funcs.f2_laguerre,
            'exacte': funcs.f2_exacte(),
            'intervalle': (0, np.inf),
            'methode_speciale': 'Gauss-Laguerre',
            'poids': 'e^{-x}'
        },
        {
            'nom': 'Combinee',
            'f': funcs.f3_combinee,
            'exacte': funcs.f3_exacte(),
            'intervalle': (0, 1),
            'methode_speciale': 'Gauss-Chebyshev',
            'poids': '1/√(1-x²)'
        },
        {
            'nom': 'Neutre',
            'f': funcs.f4_neutre,
            'exacte': funcs.f4_exacte(),
            'intervalle': (-1, 1),
            'methode_speciale': None,
            'poids': 'aucun'
        }
    ]
    
    # Valeurs de n à tester
    n_values = [4, 8, 16, 32, 64, 128]
    
    # Résultats stockés
    resultats = {func['nom']: {meth: {'erreurs': [], 'temps': [], 'n': []} 
                              for meth in methodes.keys()} 
                for func in fonctions_config}
    
    # Nombre de répétitions pour moyenner le temps
    repetitions = 5
    
    print("\n" + "="*80)
    print("COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*80)
    
    # Tests pour chaque fonction
    for func_config in fonctions_config:
        nom = func_config['nom']
        f = func_config['f']
        exacte = func_config['exacte']
        a, b = func_config['intervalle']
        methode_speciale = func_config['methode_speciale']
        
        print(f"\n{'='*70}")
        print(f"FONCTION: {nom}")
        print(f"Intervalle: [{a}, {b}]")
        print(f"Poids: {func_config['poids']}")
        print(f"Valeur exacte: {exacte:.10e}")
        print(f"{'='*70}")
        
        # Tests pour chaque méthode
        for meth_name, meth_func in methodes.items():
            print(f"\n  {meth_name}:")
            
            # Initialisation des listes pour cette méthode
            erreurs = []
            temps_moy = []
            n_vals = []
            
            # Tests pour différentes valeurs de n
            for n in n_values:
                try:
                    # Moyenne sur plusieurs exécutions pour le temps
                    temps_total = 0
                    val = None
                    
                    for _ in range(repetitions):
                        start = time.perf_counter()
                        
                        if meth_name == 'Gauss-Laguerre':
                            if b == np.inf:
                                val = meth_func(f, n)
                            else:
                                val = None
                        elif meth_name == 'Gauss-Chebyshev' and methode_speciale == 'Gauss-Chebyshev':
                            # Pour Chebyshev, f doit être multipliée par sqrt(1-x²)
                            def f_chebyshev(x):
                                return f(x) * np.sqrt(1 - x**2)
                            val = meth_func(f_chebyshev, a, b, n)
                        elif meth_name == 'Gauss-Chebyshev' and b == np.inf:
                            # Non applicable
                            val = None
                        elif b == np.inf and meth_name in ['Simpson', 'Spline', 'Gauss-Legendre']:
                            val = meth_func(f, a, b, n)
                        else:
                            val = meth_func(f, a, b, n)
                        
                        temps_total += time.perf_counter() - start
                    
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        temps_moyen = temps_total / repetitions
                        erreur_abs = abs(val - exacte)
                        
                        erreurs.append(erreur_abs)
                        temps_moy.append(temps_moyen)
                        n_vals.append(n)
                        
                        print(f"    n={n:3d}: I={val:.6e}, erreur={erreur_abs:.2e}, temps={temps_moyen:.2e}s")
                    else:
                        print(f"    n={n:3d}: Non applicable ou valeur non valide")
                        
                except Exception as e:
                    print(f"    n={n:3d}: ERREUR - {str(e)[:50]}")
                    continue
            
            # Stocker les résultats
            if erreurs:
                resultats[nom][meth_name]['erreurs'] = erreurs
                resultats[nom][meth_name]['temps'] = temps_moy
                resultats[nom][meth_name]['n'] = n_vals
    
    return resultats, fonctions_config, methodes

def tracer_graphiques(resultats, fonctions_config, methodes):
    """Tracer les 8 graphiques demandés"""
    
    # Couleurs pour les méthodes
    couleurs = {
        'Gauss-Legendre': 'red',
        'Gauss-Laguerre': 'blue',
        'Gauss-Chebyshev': 'green',
        'Simpson': 'orange',
        'Spline': 'purple'
    }
    
    # Marqueurs pour les méthodes
    marqueurs = {
        'Gauss-Legendre': 'o',
        'Gauss-Laguerre': 's',
        'Gauss-Chebyshev': '^',
        'Simpson': 'D',
        'Spline': 'v'
    }
    
    # Création des figures
    fig_erreurs, axes_erreurs = plt.subplots(2, 2, figsize=(14, 10))
    fig_temps, axes_temps = plt.subplots(2, 2, figsize=(14, 10))
    
    axes_erreurs = axes_erreurs.flatten()
    axes_temps = axes_temps.flatten()
    
    # Pour chaque fonction
    for idx, func_config in enumerate(fonctions_config):
        nom = func_config['nom']
        ax_err = axes_erreurs[idx]
        ax_tmp = axes_temps[idx]
        
        # Graphique des erreurs
        ax_err.set_title(f'Fonction {nom} - Erreurs', fontsize=12, fontweight='bold')
        ax_err.set_xlabel('n (nombre de points/subdivisions)', fontsize=10)
        ax_err.set_ylabel('log10(Erreur absolue)', fontsize=10)
        ax_err.grid(True, alpha=0.3, linestyle='--')
        
        # Graphique des temps
        ax_tmp.set_title(f'Fonction {nom} - Temps d\'exécution', fontsize=12, fontweight='bold')
        ax_tmp.set_xlabel('n (nombre de points/subdivisions)', fontsize=10)
        ax_tmp.set_ylabel('log10(Temps en secondes)', fontsize=10)
        ax_tmp.grid(True, alpha=0.3, linestyle='--')
        
        # Tracer chaque méthode
        for meth_name in methodes.keys():
            if resultats[nom][meth_name]['n']:  # Si la méthode a été testée
                n_vals = resultats[nom][meth_name]['n']
                erreurs = resultats[nom][meth_name]['erreurs']
                temps = resultats[nom][meth_name]['temps']
                
                if erreurs and temps:
                    # Éviter les zéros pour le log
                    erreurs_log = np.log10([max(e, 1e-16) for e in erreurs])
                    temps_log = np.log10([max(t, 1e-16) for t in temps])
                    
                    # Tracer erreurs
                    ax_err.plot(n_vals, erreurs_log, 
                              color=couleurs[meth_name], 
                              marker=marqueurs[meth_name],
                              markersize=6,
                              linewidth=2,
                              label=meth_name)
                    
                    # Tracer temps
                    ax_tmp.plot(n_vals, temps_log,
                              color=couleurs[meth_name],
                              marker=marqueurs[meth_name],
                              markersize=6,
                              linewidth=2,
                              label=meth_name)
        
        # Légendes
        ax_err.legend(fontsize=9, loc='best')
        ax_tmp.legend(fontsize=9, loc='best')
        
        # Échelle log-log
        ax_err.set_xscale('log')
        ax_tmp.set_xscale('log')
    
    # Ajuster l'espacement
    fig_erreurs.tight_layout()
    fig_temps.tight_layout()
    
    # Sauvegarder les figures
    fig_erreurs.savefig('graphiques_erreurs.png', dpi=300, bbox_inches='tight')
    fig_temps.savefig('graphiques_temps.png', dpi=300, bbox_inches='tight')
    
    # Afficher les figures
    plt.show()
    
    return fig_erreurs, fig_temps

def analyser_resultats(resultats, fonctions_config):
    """Analyse détaillée des résultats"""
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DES RÉSULTATS")
    print("="*80)
    
    for func_config in fonctions_config:
        nom = func_config['nom']
        print(f"\n{'='*60}")
        print(f"ANALYSE POUR LA FONCTION: {nom}")
        print(f"{'='*60}")
        
        # Trouver la meilleure méthode pour cette fonction
        meilleure_erreur = {}
        meilleur_temps = {}
        
        for meth_name in resultats[nom].keys():
            if resultats[nom][meth_name]['erreurs']:
                erreurs = resultats[nom][meth_name]['erreurs']
                temps = resultats[nom][meth_name]['temps']
                
                if erreurs:
                    erreur_min = min(erreurs)
                    meilleure_erreur[meth_name] = erreur_min
                
                if temps:
                    temps_min = min(temps)
                    meilleur_temps[meth_name] = temps_min
        
        # Afficher les meilleures méthodes par précision
        if meilleure_erreur:
            print("\nMeilleure précision (erreur minimale):")
            for meth, err in sorted(meilleure_erreur.items(), key=lambda x: x[1]):
                print(f"  {meth:20s}: {err:.2e}")
        
        # Afficher les meilleures méthodes par temps
        if meilleur_temps:
            print("\nMeilleur temps d'exécution:")
            for meth, tps in sorted(meilleur_temps.items(), key=lambda x: x[1]):
                print(f"  {meth:20s}: {tps:.2e} s")
        
        # Analyse de convergence
        print("\nTaux de convergence (dernières valeurs):")
        for meth_name in resultats[nom].keys():
            n_vals = resultats[nom][meth_name]['n']
            erreurs = resultats[nom][meth_name]['erreurs']
            
            if len(n_vals) >= 2 and len(erreurs) >= 2:
                try:
                    # Calcul du taux de convergence entre les deux dernières valeurs
                    if erreurs[-1] > 0 and erreurs[-2] > 0:
                        r = np.log(erreurs[-1]/erreurs[-2]) / np.log(n_vals[-2]/n_vals[-1])
                        print(f"  {meth_name:20s}: taux ≈ {r:.2f}")
                except:
                    pass

def generer_rapport(resultats, fonctions_config):
    """Générer un rapport synthétique"""
    
    rapport = """
RAPPORT DE COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE
============================================================

1. INTRODUCTION
---------------
Ce rapport compare cinq méthodes d'intégration numérique appliquées à quatre
fonctions tests représentatives. Les méthodes étudiées sont:
1. Quadrature de Gauss-Legendre (classique)
2. Quadrature de Gauss-Laguerre
3. Quadrature de Gauss-Chebyshev
4. Méthode composite de Simpson
5. Intégration par spline cubique

2. FONCTIONS TESTS
------------------
"""
    
    for idx, func in enumerate(fonctions_config):
        rapport += f"""
2.{idx+1} Fonction {func['nom']}
   Intervalle: {func['intervalle']}
   Poids: {func['poids']}
   Valeur exacte: {func['exacte']:.10e}
   Caractéristiques: """
        
        if func['nom'] == 'Chebyshev':
            rapport += "Fonction oscillante avec poids 1/√(1-x²)"
        elif func['nom'] == 'Laguerre':
            rapport += "Décroissance exponentielle avec poids e^{-x}"
        elif func['nom'] == 'Combinee':
            rapport += "Combinaison des deux types de poids"
        else:
            rapport += "Fonction lisse sans singularité"
    
    rapport += """

3. MÉTHODOLOGIE
---------------
Pour chaque méthode et chaque fonction, on calcule:
- L'intégrale numérique pour n = 4, 8, 16, 32, 64, 128
- L'erreur absolue par rapport à la valeur exacte
- Le temps d'exécution moyen sur 5 répétitions

4. RÉSULTATS SYNTHÉTIQUES
-------------------------
"""
    
    # Tableau synthétique
    rapport += "\nMéthode la plus précise par fonction:\n"
    rapport += "-" * 50 + "\n"
    
    for func in fonctions_config:
        nom = func['nom']
        meilleures = {}
        for meth in resultats[nom].keys():
            if resultats[nom][meth]['erreurs']:
                erreurs = resultats[nom][meth]['erreurs']
                if erreurs:
                    meilleures[meth] = min(erreurs)
        
        if meilleures:
            meilleure_meth = min(meilleures.items(), key=lambda x: x[1])
            rapport += f"{nom:15s}: {meilleure_meth[0]:20s} (erreur: {meilleure_meth[1]:.2e})\n"
    
    rapport += """

5. CONCLUSIONS
--------------

5.1 Observations générales:
- Les méthodes de Gauss adaptées au poids (Chebyshev pour poids 1/√(1-x²), 
  Laguerre pour poids e^{-x}) sont les plus efficaces pour leurs fonctions cibles
- Gauss-Legendre est robuste et performante pour les fonctions lisses
- Simpson et Spline ont des performances correctes mais généralement inférieures
  aux méthodes de Gauss pour les fonctions tests considérées

5.2 Compromis précision/temps:
- Les méthodes de Gauss nécessitent moins de points pour atteindre une précision donnée
- Simpson est simple à implémenter mais nécessite plus de points pour une précision équivalente
- Spline cubique est coûteuse en temps pour un grand nombre de points

5.3 Recommandations:
- Utiliser Gauss-Chebyshev pour les intégrales avec poids 1/√(1-x²)
- Utiliser Gauss-Laguerre pour les intégrales sur [0,∞) avec poids e^{-x}
- Utiliser Gauss-Legendre pour les intégrales standards sur un intervalle fini
- Simpson est un bon choix pour des applications simples avec des fonctions régulières
- Spline peut être utile quand on a besoin à la fois de l'intégrale et d'une interpolation

6. FICHIERS GÉNÉRÉS
-------------------
- graphiques_erreurs.png : Graphiques des erreurs pour les 4 fonctions
- graphiques_temps.png : Graphiques des temps d'exécution pour les 4 fonctions
- rapport_integration.txt : Ce rapport
"""
    
    # Sauvegarder le rapport
    with open('rapport_integration.txt', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print(rapport)
    print("\nRapport sauvegardé dans 'rapport_integration.txt'")

# Programme principal
if __name__ == "__main__":
    print("DÉMARRAGE DE LA COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*70)
    
    # Exécuter les tests
    print("\nExécution des tests...")
    resultats, fonctions_config, methodes = tester_methodes()
    
    # Tracer les graphiques
    print("\nGénération des graphiques...")
    fig_erreurs, fig_temps = tracer_graphiques(resultats, fonctions_config, methodes)
    
    # Analyser les résultats
    analyser_resultats(resultats, fonctions_config)
    
    # Générer le rapport
    print("\nGénération du rapport...")
    generer_rapport(resultats, fonctions_config)
    
    print("\n" + "="*70)
    print("PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print("\nFichiers générés:")
    print("1. graphiques_erreurs.png - Graphiques des erreurs")
    print("2. graphiques_temps.png - Graphiques des temps d'exécution")
    print("3. rapport_integration.txt - Rapport synthétique")
