"""Sequential authorized training, then sealed final evaluation."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    for name, arguments in (("runner.py", ["train"]), ("final_reporting.py", [])):
        subprocess.run([sys.executable, "-u", str(HERE / name), *arguments],
                       cwd=HERE.parents[2], check=True)


if __name__ == "__main__":
    main()
