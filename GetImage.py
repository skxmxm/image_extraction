from tkinter import Tk
from tkinter.filedialog import askopenfilename


def get_image():
    Tk().withdraw()
    filename = askopenfilename()
    return filename
