import torch

class CattleNet:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

    def detect(self, img):
        results = self.model(img)
        return results.pandas().xyxy[0].to_dict("records")
