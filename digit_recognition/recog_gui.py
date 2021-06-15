from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab
import numpy as np

model = load_model('mnist.h5')


def predict_digit(img):
    # Resize image to 28x28
    img = img.resize((28, 28))

    # Convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)

    # Reshape to fit model input
    img = img.reshape(1, 28, 28, 1)

    # Predicting the class
    pred = model.predict(img)
    return np.argmax(pred), np.amax(pred)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title('Recognise Digit GUI')
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=0, pady=2)
        self.button_clear.grid(row=1, column=1, pady=2, padx=2)

        self.canvas.bind('<B1-Motion>', self.draw_lines)

    def clear_all(self):
        self.canvas.delete('all')

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()           # Get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)     # Get the coordinate of the canvas
        x1, y1, x2, y2 = rect
        rect = (x1 + 42, y1 + 42, x2 + 102, y2 + 102)
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(round(acc * 100, 2)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        radius = 8
        self.canvas.create_oval(self.x - radius, self.y - radius,
                                self.x + radius, self.y + radius, fill='black')


app = App()
mainloop()
