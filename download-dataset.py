
from roboflow import Roboflow

rf = Roboflow(api_key="hUaMASK3d1LT1z3y8as3")
project = rf.workspace("realsoftai").project("bozor-classification-2-lzxzd")
version = project.version(8)
dataset = version.download("folder")


