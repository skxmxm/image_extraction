from GetImage import get_image
from ProcessImage import ProcessImage

def back_end(pic):
    pi = ProcessImage(pic)
    return pi.start_processing()