import cv2 as cv
import numpy as np
import traceback
import GetCmd
import time
import os


class ProcessImage():
    BLUE =  [255, 0, 0]
    RED =   [0, 0, 255]
    GREEN = [0, 255, 0]
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]

    DRAW_BG =       {'color': BLACK, 'val': 0}
    DRAW_FG =       {'color': WHITE, 'val': 1}
    DRAW_PR_BG =    {'color': RED, 'val': 2}
    DRAW_PR_FG =    {'color': GREEN, 'val': 3}

    # setting up flags
    rect = (0, 0, 1, 1)
    drawing =   False
    rectangle = False
    rect_over = False
    rect_or_mask = 100
    value = DRAW_FG
    thickness = 3

    def __init__(self, filename):
        self.img = self.img = cv.imread(cv.samples.findFile(filename))
        self.img_cp = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)
        self.filename = filename
        self.result = None

        # input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.drawmask)
        cv.moveWindow('output', x=0, y=0)
        cv.moveWindow('input', x=500, y=300)

    def restore_image(self):
        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rectangle = False
        self.rect_or_mask = 100
        self.rect_over = False
        self.value = self.DRAW_FG
        self.img = self.img_cp.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)

    def update_image(self):
        try:
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            if self.rect_or_mask == 0:      # rect mode
                cv.grabCut(self.img_cp, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                self.rect_or_mask = 1
            elif self.rect_or_mask == 1:    # mask mode
                cv.grabCut(self.img_cp, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
        except:
            traceback.print_exc()

    def save_image(self):
        path = os.path.dirname(self.filename)
        print("Your file has been saved to: " + path)
        target = path + '/output.png'
        cv.imwrite(target, self.output)
        return target

    def start_processing(self):
        while (1):
            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            
            cmd = GetCmd.get_cmd_from_terminal()

            if cmd == 'esc':
                self.result = self.save_image()
                cv.destroyAllWindows()
                break
            elif cmd == 'choose sure background':
                self.value = self.DRAW_BG
            elif cmd == 'choose sure foreground':
                self.value = self.DRAW_FG
            elif cmd == 'choose probable background':
                self.value = self.DRAW_PR_BG
            elif cmd == 'choose probable foreground':
                self.value = self.DRAW_PR_FG
            elif cmd == 'restore image':
                self.restore_image()
            elif cmd == 'update image':
                for i in range(3):
                    time.sleep(0.01)
                    self.update_image()

            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.img_cp, self.img_cp, mask=mask2)
        return self.result

    def drawmask(self, event, x, y, flags, param):
        # rect mode
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y
        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img_cp.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0
        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0

        # touchup mode
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
            cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
