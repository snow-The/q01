"""讓 tests/ 在未安裝套件時也能 import q01（CI 與本機皆同）。"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
