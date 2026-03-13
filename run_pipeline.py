from pathlib import Path
from pipeline.sotif_pipeline import SOTIFPipeline

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    pipeline = SOTIFPipeline(base_dir)
    pipeline.run()
