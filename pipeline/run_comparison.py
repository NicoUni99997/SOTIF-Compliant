from clustering import run_clustering
from scores import compute_scores
from graphs import draw_graphs


def main():
    print("\n=== Avvio comparazione scenari ===\n")

    '''print("[1/3] Clustering scenari in corso...")
    run_clustering()
    print("[OK] Clustering completato.\n")'''

    print("[2/3] Calcolo punteggi...")
    compute_scores()
    print("[OK] Punteggi calcolati.\n")

    print("[3/3] Generazione grafici...")
    draw_graphs()
    print("[OK] Grafici salvati.\n")

    print("=== Pipeline completata con successo ===")


if __name__ == "__main__":
    main()
