### pip install torchinfo 필요
from torchinfo import summary
import torch
from ultralytics import YOLO

def visualize_yolo_structure(model):
    """YOLO 구조를 시각적으로 표현"""
    
    backbone_layers = []
    neck_layers = []
    head_layers = []
    
    for i, layer in enumerate(model.model.model):
        layer_info = f"L{i}: {type(layer).__name__}"
        
        if i <= 10:  # backbone (조정 필요할 수 있음)
            backbone_layers.append(layer_info)
        elif i <= 19:  # neck
            neck_layers.append(layer_info)
        else:  # head
            head_layers.append(layer_info)
    
    print("┌─ YOLO v8 Architecture ─┐")
    print("│                        │")
    print("│ 🔷 BACKBONE:           │")
    for layer in backbone_layers:
        print(f"│   {layer:<20} │")
    print("│                        │")
    print("│ 🔸 NECK:               │")
    for layer in neck_layers:
        print(f"│   {layer:<20} │")
    print("│                        │")
    print("│ 🔹 HEAD:               │")
    for layer in head_layers:
        print(f"│   {layer:<20} │")
    print("│                        │")
    print("└────────────────────────┘")

yoloYAMLs = ('yolov8n.yaml', 'yolov8s.yaml','yolov8m.yaml', 'yolov8l.yaml', 'yolov8x.yaml')

# pre-trained model : *.pt --> YOLO('yolov8n.pt') 실행하면 최초 1 번은 YOLO site에서 download 받음.
#yoloModels = ('yolov8n.pt', 'yolov8s.pt','yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')

for yml in yoloYAMLs : 

    model = YOLO (yml)
    result = model.info()
    print (f"YOLO Model {yml} : {result}")

    #print (model.model)
    #summary(model.model)
    visualize_yolo_structure(model)