import tkinter as tk
from tkinter import filedialog
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image, ImageTk

# Initialize the model and other components as you did in your code
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds, image

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_caption, input_image = predict_step(file_path)
        result_label.config(text="Predicted Caption: " + image_caption[0])
        display_image(input_image)

def display_image(image):
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img

# Create the GUI window
window = tk.Tk()
window.title("Image Captioning App")

# Create a browse button
browse_button = tk.Button(window, text="Browse Image", command=browse_image)
browse_button.pack()

# Create a label for the result
result_label = tk.Label(window, text="Predicted Caption: ", font=("Times New Roman", 20))
result_label.pack()

# Create a label for the input image
image_label = tk.Label(window)
image_label.pack()

# Start the GUI main loop
window.mainloop()
