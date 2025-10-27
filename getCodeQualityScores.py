import os
import re
from pylint import epylint as lint
import pycodestyle
import subprocess


def get_pylint_score(file_path):
    """Run pylint on a file and return its numeric score."""
    command = ["pylint", file_path]
    result = subprocess.run(command, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # print("STDOUT:\n", stdout)
    # print("STDERR:\n", stderr)
    # print("Return code:", returncode)

    # Extract score from stdout
    matches = re.findall(r"Your code has been rated at ([0-9\.]+)/10", stdout)
    if matches:
        return float(matches[-1])
    return 0.0


def get_pep8_score(file_path):
    """Run pycodestyle on a file and compute a simple PEP8 score."""
    style = pycodestyle.StyleGuide(quiet=True)
    report = style.check_files([file_path])
    total_errors = getattr(report, 'total_errors', 0)
    # Convert error count into a score between 0 and 10 (custom heuristic)
    score = max(0, 10 - total_errors / 10)
    return round(score, 2), total_errors


def analyze_directory(directory, recursive=False):
    """Loop over Python files in a directory and print their quality scores."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                pylint_score = get_pylint_score(file_path)
                pep8_score, pep8_errors = get_pep8_score(file_path)

                print(f"{file_path} \t Pylint Score: {pylint_score:.2f}/10 \t PEP8 Score: {pep8_score:.2f}/10")
                # print(f"  Pylint Score: {pylint_score:.2f}/10")
                # print(f"  PEP8 Score:   {pep8_score:.2f}/10 ({pep8_errors} issues)")
                # print("-" * 60)

        if not recursive:
            break


if __name__ == "__main__":
    # Change this path to your target folder
    target_directory = "./First experiment/"
    analyze_directory(target_directory, recursive=True)

    print('*****************************')
    target_directory = "./Second experiment/"
    analyze_directory(target_directory, recursive=True)

    print('*****************************')
    target_directory = "./Third experiment/"
    analyze_directory(target_directory, recursive=True)
    print('*****************************')
