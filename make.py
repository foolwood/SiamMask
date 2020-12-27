import os

abspath = os.path.abspath(__file__)
projectDir = os.path.dirname(abspath)

pyVotkitDir = "/utils/pyvotkit"
pySotDir = "/utils/pysot/utils"

os.chdir(projectDir + pyVotkitDir)
os.system("python setup.py build_ext --inplace")

os.chdir(projectDir + pySotDir)
os.system("python setup.py build_ext --inplace")

os.chdir(projectDir)