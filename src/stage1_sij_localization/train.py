import os
from ultralytics import YOLO


os.environ['WANDB_MODE'] = 'online' 

model = YOLO('yolo12x.pt')  

model.train(
    data='bbox_detection/dataset_oneclass_no_gaussian_20250701/data.yaml',  
    epochs=500,
    imgsz=512,
    batch=64,
    device='0,1,2,3',
    project='runs/detect',
    name='yolo12x-singleclass-no-gaussian-20250701',
    save=True,
    save_period=-1,    
    verbose=True,
    patience=50,
    auto_augment='randaugment'
)
