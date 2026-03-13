import logging
import subprocess
from pathlib import Path


class SOTIFPipeline:
    """
    Pipeline completa SOTIF-like (multi-dataset):
    Per ogni cartella dentro base_dir/datasets:
    1. Sanity check log base
    2. Calcolo ODD (descrittivo)
    3. Calcolo Hazard (Leaderboard-based)
    4. Generazione report finale
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

        # Directory datasets (ora è la fonte dinamica)
        self.datasets_dir = base_dir / "datasets"

        # Script già esistenti
        self.stepA_odd_script = base_dir / "data_gathering" / "enriching" / "compute_sotif_odd.py"
        self.stepB_hazard_script = base_dir / "data_gathering" / "enriching" / "compute_sotif_hazard.py"
        self.stepC_report_script = base_dir / "data_gathering" / "enriching" / "compute_sotif.py"
        self.stepB2_feature_script = base_dir / "data_gathering" / "enriching" / "compute_feature_vectors.py"

        self._setup_logger()

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger("SOTIFPipeline")

    # --------------------------------------------------
    # Utility per eseguire script (con argomento dataset_dir)
    # --------------------------------------------------
    def _run_script(self, script: Path, step_name: str, dataset_dir: Path):
        if not script.exists():
            raise FileNotFoundError(f"{step_name} script non trovato: {script}")

        self.logger.info(f"▶ Avvio {step_name} | dataset: {dataset_dir.name}")
        result = subprocess.run(
            ["python", str(script), "--dataset_dir", str(dataset_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.logger.error(f"✖ Errore in {step_name} | dataset: {dataset_dir.name}")
            if result.stderr:
                self.logger.error(result.stderr)
            raise RuntimeError(f"{step_name} fallito per {dataset_dir.name}")

        self.logger.info(f"✔ {step_name} completato | dataset: {dataset_dir.name}")
        if result.stdout:
            self.logger.debug(result.stdout)

    # --------------------------------------------------
    # Scansione datasets
    # --------------------------------------------------
    def list_dataset_folders(self):
        if not self.datasets_dir.exists():
            raise FileNotFoundError(f"Cartella datasets non trovata: {self.datasets_dir}")

        folders = [p for p in self.datasets_dir.iterdir() if p.is_dir()]
        if not folders:
            raise RuntimeError(f"Nessuna sottocartella trovata in {self.datasets_dir}")

        self.logger.info(f"Trovate {len(folders)} cartelle dataset in: {self.datasets_dir}")
        return sorted(folders)

    # --------------------------------------------------
    # STEP 0 – Sanity check logs (per dataset)
    # --------------------------------------------------
    def check_logs(self, dataset_dir: Path):
        self.logger.info(f"Controllo presenza log base in: {dataset_dir}")

        # Se i log stanno direttamente nella cartella dataset:
        logs = list(dataset_dir.glob("*_log_basic.json"))

        # Se vuoi cercare anche in sottocartelle, usa invece:
        # logs = list(dataset_dir.rglob("*_log_basic.json"))

        if not logs:
            raise RuntimeError(f"Nessun log base trovato in {dataset_dir}. Pipeline interrotta per questo dataset.")

        self.logger.info(f"Trovati {len(logs)} log base in {dataset_dir.name}.")
        return logs

    # --------------------------------------------------
    # STEP 1 – Arricchimento + ODD
    # --------------------------------------------------
    def compute_odd(self, dataset_dir: Path):
        self._run_script(
            self.stepA_odd_script,
            "STEP A – Analisi ODD (descrittiva)",
            dataset_dir
        )

    # --------------------------------------------------
    # STEP 2 – Hazard + probabilità + severità
    # --------------------------------------------------
    def compute_hazard(self, dataset_dir: Path):
        self._run_script(
            self.stepB_hazard_script,
            "STEP B – Hazard & Severity (Leaderboard)",
            dataset_dir
        )

    # --------------------------------------------------
    # STEP 3 – Future Vectors to UMAP and K-Means
    # --------------------------------------------------
    def compute_feature_vectors(self, dataset_dir: Path):
        self._run_script(
            self.stepB2_feature_script,
            "STEP B2 – Estrazione Feature Vectors (RQ3)",
            dataset_dir
        )

    # --------------------------------------------------
    # STEP 4 – Report finale SOTIF
    # --------------------------------------------------
    def compute_final_report(self, dataset_dir: Path):
        self._run_script(
            self.stepC_report_script,
            "STEP C – Report SOTIF finale",
            dataset_dir
        )

    # --------------------------------------------------
    # Pipeline completa (per tutti i dataset)
    # --------------------------------------------------
    def run(self):
        self.logger.info("===== AVVIO PIPELINE SOTIF (MULTI-DATASET) =====")

        dataset_folders = self.list_dataset_folders()

        for dataset_dir in dataset_folders:
            self.logger.info(f"\n===== DATASET: {dataset_dir.name} =====")
            self.check_logs(dataset_dir)
            self.compute_odd(dataset_dir)
            self.compute_hazard(dataset_dir)
            self.compute_final_report(dataset_dir)

        self.logger.info("\n===== PIPELINE SOTIF COMPLETATA =====")
        self.logger.info(f"Dataset finali disponibili in: {self.datasets_dir}")
