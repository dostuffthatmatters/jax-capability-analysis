import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_type_annotations() -> None:
    exit_code = os.system(f"cd {PROJECT_DIR} && mypy main.py")
    assert exit_code == 0
