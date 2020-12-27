import sys
import os
from shutil import copyfile
VOT_YEAR = sys.argv[1]
VOT_CHALLENGE = "main"

defaultName = f"vot{VOT_YEAR}_{VOT_CHALLENGE}"

dl = "../" + sys.argv[2]
data = "../../" + sys.argv[3]
# scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

try:
    os.makedirs(data)
except FileExistsError:
    print("Loading...")

descriptionFileSrc = f"../dl/{defaultName}/description.json"
descriptionFileDestination = f"{data}/description.json"
copyfile(descriptionFileSrc, descriptionFileDestination)

os.system(f"python unzip_vot.py {dl} {data}")