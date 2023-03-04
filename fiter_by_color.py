import cv2
import numpy as np

def filter_by_color(img, box):
    to_filter = True

    white_lower = (52, 78, 0)
    white_upper = (255, 255, 255)

    blue_lower = (0, 0, 0)
    blue_upper = (92, 255, 255)

    green_lower = (0, 128, 67)
    green_upper = (77, 255, 255)

    yellow_lower = (38, 0, 0)
    yellow_upper = (255, 255, 255)

    red_lower = (15, 0, 0)
    red_upper = (255, 255, 255)

    pink_lower = (37, 0, 0)
    pink_upper = (168, 255, 255)


    brown_lower = (40, 0, 0)
    brown_upper = (255, 255, 255)

    black_lower = (10, 87, 63)
    black_upper = (255, 255, 255)

    patch = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
    path_col = np.mean(patch , axis=(0,1))
    colors = ('white', 'blue', 'green', 'yellow', 'red', 'pink', 'brown', 'black')
    for col in colors:
        str_lower = f'{col}_lower'
        str_upper = f'{col}_upper'
        if path_col[0] >= eval(str_lower)[0] and path_col[0] <= eval(str_upper)[0] and\
           path_col[1] >= eval(str_lower)[1] and path_col[1] <= eval(str_upper)[1] and\
           path_col[2] >= eval(str_lower)[2] and path_col[2] <= eval(str_upper)[2]:
           to_filter = False
           # print( f' Ball is {col}')
           break
    return to_filter
