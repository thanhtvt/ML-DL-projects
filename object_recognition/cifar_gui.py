from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np


def predict(path):
    # Resize image to 32x32
    img = load_img(path, target_size=(32, 32))

    # Convert to array
    img = img_to_array(img)

    # Reshape to fit model input
    img = img.reshape(1, 32, 32, 3)

    # Normalize data
    img = img.astype('float32')
    img = img / 255.0

    # Predicting the class
    pred = model.predict(img)
    return np.argmax(pred), np.amax(pred)


def get_image():
    # Get path
    path = askopenfilename(filetype=(("jpg file", "*.jpg"), ("png file", '*.png'), ("All files", "*.*"),))

    # Load image
    global img
    img = Image.open(path)
    img = img.resize((350, 350))
    img = ImageTk.PhotoImage(img)

    return path, img


def convert_class_to_str(result):
    if result == 0:
        class_str = 'airplane'
    elif result == 1:
        class_str = 'automobile'
    elif result == 2:
        class_str = 'bird'
    elif result == 3:
        class_str = 'cat'
    elif result == 4:
        class_str = 'deer'
    elif result == 5:
        class_str = 'dog'
    elif result == 6:
        class_str = 'frog'
    elif result == 7:
        class_str = 'horse'
    elif result == 8:
        class_str = 'ship'
    else:
        class_str = 'truck'
    return class_str


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title('Cifar10 Classification GUI')
        self.geometry('575x420')

        # Creating elements
        self.text = tk.Entry(self, width=25, font=('Helvetica', 12))
        self.add_btn = tk.Button(self, text='Browse', command=self.add_image)
        self.classify_btn = tk.Button(self, text='Classify', command=self.classify_image)
        self.clear_btn = tk.Button(self, text="Erase", command=self.delete_image)
        self.canvas = tk.Canvas(self, width=350, height=350, bg='white')
        self.label_pred = tk.Label(self, text='Prediction\nhere', font=('Helvetica', 20))

        # Structure
        self.text.grid(row=0, column=0, pady=2)
        self.add_btn.grid(row=0, column=1, pady=2, padx=2)
        self.canvas.grid(row=1, column=0, pady=2, sticky=W, )
        self.label_pred.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=2, column=0, pady=2)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)

    def add_image(self):
        # Erase past record
        self.delete_image()

        # Load image
        path, img = get_image()

        # Display path and image
        self.text.insert(END, path)
        self.canvas.create_image(0, 0, anchor=NW, image=img)

    def classify_image(self):
        # Load image
        filepath = self.text.get()

        res, acc = predict(filepath)
        class_res = convert_class_to_str(res)
        self.label_pred.configure(text=class_res + ', ' + str(round(acc * 100, 2)) + '%')

    def delete_image(self):
        self.canvas.delete('all')
        self.text.delete(0, END)


if __name__ == '__main__':
    model = load_model('cifar10.h5')
    app = App()
    mainloop()
