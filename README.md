# Verso la validazione automatica: un'analisi empirica di strumenti di generazione di scenari di guida

Questa repository costituisce la base del lavoro di tesi triennale, finalizzato a un’analisi comparativa di diversi tool di generazione di scenari di test per sistemi di guida autonoma (ADS).
L’analisi viene condotta lungo tre dimensioni complementari:

1. **Criticità** ovvero la capacità degli scenari di evidenziare situazioni pericolose o al limite.

2. **Diversità** ossia il grado di varietà e copertura delle condizioni generate.

3. **Efficacia** intesa come la capacità di concentrare gli scenari critici in grado di mettere in crisi l’ADS.
---
## Struttura della repository

```
│
├── clustering_results     # Risultati clustering degli scenari
│   ├── clustering_metrics_comparison.svg
│   ├── clustering_results.json
│   └── plotting           # Grafici visualizzazione clusters
│       └── clusters.svg
│
├── datasets               # Dataset generati per l’analisi
│   ├── critical_dataset.csv
│   ├── full_dataset.csv
│   └── full_dataset_with_clusters.csv
│
├── data_gathering         # Moduli per logging e arricchimento
│   ├── carlaBasicLogger.py
│   ├── violationMonitor.py
│   └── enriching           # Moduli di arricchimento
│       ├── critical.py
│       ├── dynamic.py
│       ├── functional.py
│       └── orchestrator.py
│
├── pipeline               # Moduli della pipeline di analisi
│   ├── clustering.py
│   ├── graphs.py
│   ├── run_comparison.py
│   └── scores.py
│
├── PythonAPI              # API Python di CARLA 0.9.13 modificata
│   ├── python_api.md
│   └── carla
│       └── agents
│           └── navigation
│               ├── behavior_agent.py
│               ├── custom_behavior_agent.py
│               ├── stop_manager.py
│               ├── --- altri moduli ---
│
├── results                # Risultati finali delle analisi
│   ├── latent_risk_summary.csv
│   ├── summary_table.csv
│   ├── tool_scores.json
│   └── graphs
│       ├── cluster_heatmap.svg
│       ├── criticality_profile.svg
│       ├── effectiveness_barplot.svg
│       ├── latent_risk_analysis.svg
│       └── outcome_distribution.svg
│
├── stubs                  # Stub di tipizzazione CARLA
│   └── carla
│       ├── command.pyi
│       └── __init__.pyi
│
├── utils                  # Moduli helper
│   ├── carla_help.py
│   └── json_help.py
│
├── docs                   # Documentazione varia
│   ├── enrichment.md
│   ├── file_structure.md
│   ├── integration.md
│   ├── pipeline.md
│
├── requirements.txt       # Dipendenze Python
```
---
## Startup
### Dipendenze necessarie
Per utilizzare la repository è necessario creare un ambiente virtuale con **Python 3.8** e installare le dipendenze richieste tramite il comando:
```bash
pip install -r requirements.txt
```

### Utilizzo del framework per nuove analisi

Per sfruttare il framework su dati o scenari diversi da quelli forniti, fare riferimento alla 
[guida all'integrazione](docs/integration.md).

