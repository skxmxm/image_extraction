import cv2 as cv


def get_cmd_from_terminal():
    k = cv.waitKey(1)

    if k == 27:
        return 'esc'
    elif k == ord('1'):
        return 'choose sure background'
    elif k == ord('2'):
        return 'choose sure foreground'
    elif k == ord('3'):
        return 'choose probable background'
    elif k == ord('4'):
        return 'choose probable foreground'
    elif k == ord('r'):
        return 'restore image'
    elif k == ord('n'):
        return 'update image'
