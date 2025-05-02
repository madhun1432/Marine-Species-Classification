import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
import traceback

# Load the trained model
model_path = r"C:\onedrive\Desktop\project\frontend\Frontend_1\model\EFFI_NET_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    messagebox.showerror("Error", f"Model not found at {model_path}")
    raise FileNotFoundError(f"Model not found at {model_path}")

class_names = ['Crabs', 'Jelly Fish', 'Seahorse', 'Sharks', 'Starfish', 'Turtle_Tortoise']

# GUI App Class
class MarineClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marine Species Classification")
        self.root.geometry('800x800')

        self.previous_predictions = []

        self.setup_ui()

    def setup_ui(self):
        # Background image
        background_path = r"C:\onedrive\Desktop\project\frontend\Frontend_1\model\sea_bg.jpg"
        if os.path.exists(background_path):
            sea_image = Image.open(background_path)
            sea_image = sea_image.resize((800, 800))
            self.sea_bg = ImageTk.PhotoImage(sea_image)
            self.background_label = Label(self.root, image=self.sea_bg)
            self.background_label.place(relwidth=1, relheight=1)

        # Title
        Label(self.root, text="Marine Species Classification", font=("Arial", 24), bg="#80DEEA").place(relx=0.5, rely=0.05, anchor="center")

        # Image display
        self.img_label = Label(self.root, bg="lightblue", borderwidth=2, relief="solid", width=250, height=250)
        self.img_label.place(relx=0.5, rely=0.25, anchor="center")

        # Classification result
        self.result_label = Label(self.root, text="CLASSIFICATION RESULT:", font=("Arial", 16), bg="#80DEEA")
        self.result_label.place(relx=0.5, rely=0.55, anchor="center")

        # Browse Button
        self.browse_button = Button(self.root, text="Browse Image", command=self.classify_image, font=("Arial", 14), bg="#009688", fg="white")
        self.browse_button.place(relx=0.5, rely=0.65, anchor="center")

        # Reset and Exit Buttons
        Button(self.root, text="Back", command=self.reset_app, font=("Arial", 12), bg="#FF9800", fg="white").place(relx=0.4, rely=0.75, anchor="center")
        Button(self.root, text="Exit", command=self.exit_app, font=("Arial", 12), bg="#F44336", fg="white").place(relx=0.6, rely=0.75, anchor="center")

        # Prediction history label
        self.history_label = Label(self.root, text="Previous Predictions:", font=("Arial", 12), bg="#80DEEA", fg="black")
        self.history_label.place(relx=0.5, rely=0.85, anchor="center")

    def preprocess_image(self, image_path):
        image_size = (256, 256)
        img = Image.open(image_path)
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def classify_image(self):
        image_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".jpg .jpeg *.png"), ("All files", ".")]
        )

        if not image_path:
            messagebox.showinfo("Info", "No image selected.")
            return

        try:
            # Display the image
            img = Image.open(image_path)
            img.thumbnail((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk

            # Prediction
            img_array = self.preprocess_image(image_path)
            predictions = model.predict(img_array)[0]
            max_index = np.argmax(predictions)
            confidence = predictions[max_index]

            # Check if not marine
            if confidence < 0.70:
                predicted_class = "Other Species"
                confidence = 1.0
            else:
                predicted_class = class_names[max_index]

            result_text = f"Classified as: {predicted_class} (Confidence: {confidence * 100:.2f}%)"
            self.result_label.config(text=result_text)

            # Log history
            self.previous_predictions.append(result_text)
            self.update_history_label()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", "Something went wrong during classification.")

    def reset_app(self):
        self.img_label.config(image='')
        self.result_label.config(text="CLASSIFICATION RESULT:")
        self.previous_predictions = []
        self.update_history_label()

    def update_history_label(self):
        history_text = "\n".join(self.previous_predictions[-5:])  # Display only last 5 predictions
        self.history_label.config(text=f"Previous Predictions:\n{history_text}")

    def exit_app(self):
        self.root.quit()

# Launch the app
root = tk.Tk()
app = MarineClassifierApp(root)
root.mainloop()
