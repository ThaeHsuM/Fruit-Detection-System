import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# =====================
# Load CNN model
# =====================
cnn_model = tf.keras.models.load_model("fruit_cnn_model.h5")
class_labels = ['Apple', 'Banana', 'Orange']  # Same as training

# =====================
# Global variables
# =====================
start_x = start_y = end_x = end_y = 0
rect_id = None
cropping = False
img_cv = None          # Original image
img_display = None     # Resized image for display
img_tk = None

# =====================
# Mouse callbacks for cropping
# =====================
def on_mouse_down(event):
    global start_x, start_y, cropping, rect_id
    start_x, start_y = event.x, event.y
    cropping = True
    if rect_id:
        canvas.delete(rect_id)
        rect_id = None

def on_mouse_move(event):
    global rect_id
    if cropping:
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="red", width=2)

def on_mouse_up(event):
    global end_x, end_y, cropping
    end_x, end_y = event.x, event.y
    cropping = False

# =====================
# Load Image
# =====================
def open_image():
    global img_cv, img_display, img_tk
    file_path = filedialog.askopenfilename()
    if file_path:
        # Read original image
        img_cv = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        
        # Resize for display
        img_display = cv2.resize(img_cv, (640, 640))
        im_pil = Image.fromarray(img_display)
        img_tk = ImageTk.PhotoImage(image=im_pil)
        
        canvas.config(width=640, height=640)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

# =====================
# Crop & Classify
# =====================
def classify_crop():
    global start_x, start_y, end_x, end_y, img_cv, img_display
    if img_cv is None or img_display is None:
        lbl_result.config(text="No image loaded!")
        return

    # Crop coordinates on display image
    x1, y1 = min(start_x, end_x), min(start_y, end_y)
    x2, y2 = max(start_x, end_x), max(start_y, end_y)

    # Scale coordinates to original image
    h_disp, w_disp = img_display.shape[:2]
    h_orig, w_orig = img_cv.shape[:2]
    scale_x = w_orig / w_disp
    scale_y = h_orig / h_disp
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x)
    y2_orig = int(y2 * scale_y)

    crop = img_cv[y1_orig:y2_orig, x1_orig:x2_orig]
    if crop.size == 0:
        lbl_result.config(text="Invalid crop!")
        return

    # Resize to model input and normalize
    crop_resized = cv2.resize(crop, (64, 64))
    crop_norm = crop_resized / 255.0

    # Predict
    pred = cnn_model.predict(np.expand_dims(crop_norm, axis=0), verbose=0)
    label = class_labels[np.argmax(pred)]
    lbl_result.config(text=f"Predicted: {label}")

# =====================
# Tkinter GUI Setup
# =====================
root = tk.Tk()
root.title("Manual Fruit Crop + CNN Classification")

btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=5)

canvas = tk.Canvas(root)
canvas.pack()

canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

btn_classify = tk.Button(root, text="Classify Crop", command=classify_crop)
btn_classify.pack(pady=5)

lbl_result = tk.Label(root, text="Predicted: ")
lbl_result.pack(pady=5)

root.mainloop()
