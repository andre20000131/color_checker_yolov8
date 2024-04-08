import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('/home/yunxi/ai_model/ultralytics-main/ultralytics-main/colour-checker-detection-l-seg.pt')  # pretrained YOLOv8n model

im2 = cv2.imread("/home/yunxi/ai_model/data/LSMI/galaxy2/Place0/Place0_12_raw.jpg")
results = model.predict(source=im2, save=True, save_txt=True)

#print(results)


# # Define path to the image file
# source = '/home/yunxi/ai_model/data/LSMI/galaxy_512/Place2_12_gt.jpg'

# # Run inference on the source
# results = model(source)  # list of Results objects