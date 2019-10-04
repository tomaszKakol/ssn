import numpy as np
import tkinter as tk
import PIL.ImageOps
import os
from tkinter import filedialog, IntVar, StringVar
from PIL import Image, ImageTk
from tensorflow.contrib import learn
from cnn_model_fn import cnn_model_fn

window = tk.Tk()
window.title('Recognizer')
window.geometry('420x460')
window.resizable(False, False)

path_to_image = StringVar()
image_loaded = IntVar()

#----FUNCTIONS----
def adjust_picture(path):
	im = Image.open(path)
	im = im.convert("L")
	width, height = im.size
	
	if (width != 28 or height != 28):
		imm = im.resize((28, 28))
	else:
		imm = im
	
	return imm

def load_image():
	path = filedialog.askopenfilename(parent=window,title='Choose a file')
	if path:
		global image_loaded	
		image_loaded.set(1)
		
		global path_to_image
		path_to_image.set(path)
		
		im = Image.open(path)
		im = im.convert("L")
		im = im.resize((412, 412))
		im = ImageTk.PhotoImage(im)
		image_label.configure(image=im)
		image_label.image=im

def recognize():
	if (image_loaded.get() == 0):
		recognize_text.delete('0.0', tk.END)
		recognize_text.insert(tk.END, "Nie wczytano zdjecia!")
	else:
		global path_to_image
		
		im = adjust_picture(path_to_image.get())
		im = PIL.ImageOps.invert(im)
		im_np = np.array(im)
		im_np = np.float32(im_np)
		
		class_labels = ['triangle', 'circle','square','hexagon']
		
		dirname = os.path.dirname(os.path.abspath(__file__))
		shape_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir=os.path.join(dirname, 'net', 'shape_model'))
		
		predictions = shape_classifier.predict(x=im_np, as_iterable=True)
		for i, p in enumerate(predictions):
			recognize_text.delete('0.0', tk.END)
			recognize_text.tag_configure('center', justify='center')
			recognize_text.insert(tk.END, class_labels[int(p["classes"])])
			recognize_text.tag_add('center', '1.0', 'end')

#----FRAMES----

image_frame = tk.Frame(window, bg='black', relief='groove', height=420, width=420, padx=2, pady=2)
image_frame.grid(column=0, row=0, sticky="ew")
image_frame.grid_propagate(False)

button_answer_frame = tk.Frame(window, bg='green', relief='groove', height=40, width=420)
button_answer_frame.grid(column=0, row=1, sticky='ew')
button_answer_frame.grid_propagate(False)

#----LABELS-----
image_label = tk.Label(image_frame, bg='black')
image_label.grid(column=0, row=0, sticky="ew")

#----BUTTONS----
load_button = tk.Button(button_answer_frame, text='Load', height=2, width=10, command=load_image)
load_button.grid(column=0, row=0, sticky="ew")

recognize_button = tk.Button(button_answer_frame, text='Recognize', height=2, width=10, command=recognize)
recognize_button.grid(column=1, row=0, sticky="ew")

#----TEXT----
recognize_text = tk.Text(button_answer_frame, height=2, width=30, padx=8)
recognize_text.grid(column=2, row=0, sticky="ew")

window.mainloop()