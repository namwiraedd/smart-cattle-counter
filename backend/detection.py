import torch
from models.cattle_detector import CattleNet

model = CattleNet()

async def detect_cattle(file):
    results = model.detect(file.file)
    return results
