import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

# =====================
# Load CNN model
# =====================
cnn_model = tf.keras.models.load_model("fruit_cnn_model.h5")
class_labels = ['Apple', 'Banana', 'Orange']  # Same as training

# =====================
# Load YOLOv4-tiny
# =====================
yolo_config = r"C:\Detect\yolov4-tiny.cfg"
yolo_weights = r"C:\Detect\yolov4-tiny.weights"
yolo_classes_file = r"C:\Detect\coco.names"
 
net = cv2.dnn.readNet(yolo_weights, yolo_config)
with open(yolo_classes_file, "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# Fix for OpenCV version differences
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# =====================
# Helper function: detect and classify fruits
# =====================
def detect_and_classify(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # YOLO detection
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # confidence threshold
                center_x, center_y, w, h = (det[0:4] * [width, height, width, height]).astype(int)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # Draw bounding boxes and classify using CNN
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        crop = img[max(0,y):y+h, max(0,x):x+w]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (64,64))
        crop_norm = crop_resized / 255.0
        pred = cnn_model.predict(np.expand_dims(crop_norm, axis=0), verbose=0)
        label = class_labels[np.argmax(pred)]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =====================
# Tkinter GUI
# =====================
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_img = detect_and_classify(file_path)
        im_pil = Image.fromarray(result_img)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        lbl_img.config(image=imgtk)
        lbl_img.image = imgtk

root = Tk()
root.title("Fruit Detector + Classifier")

btn_open = Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=10)

lbl_img = Label(root)
lbl_img.pack()

root.mainloop()
