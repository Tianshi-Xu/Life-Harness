"""Resume authorized three-round queue; then evaluate after training is sealed."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for name, arguments in (("corrected_runner.py", ["train"]), ("corrected_final.py", [])):
        subprocess.run([sys.executable, "-u", str(HERE / name), *arguments],
                       cwd=HERE.parents[2], check=True)
