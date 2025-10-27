# from pylint import epylint as lint

# lint.py_run("./Third experiment/Claude Haiku 3.5.py", return_std=True)


import subprocess

# Command as a list of strings
command = ["pylint", "./Third experiment/Claude Haiku 3.5.py"]

# Run command and capture output
result = subprocess.run(command, capture_output=True, text=True)

# Access stdout and stderr
stdout = result.stdout
stderr = result.stderr
returncode = result.returncode

print("STDOUT:\n", stdout)
print("STDERR:\n", stderr)
print("Return code:", returncode)