"""
This script uses matching template on the thresholded image after the snooker green table removal.
The balls tracking is the same as in the color background removal script.

GENERATE_VIDEO - save the output frames in a video clip or save them separately.
"""

import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imutils.object_detection import non_max_suppression
from fiter_by_color import filter_by_color

def generator(cap):
    while cap.isOpened():
        yield


def detect_moving_balls(path_video: Path,
                        path_out: Path,
                        mask:np.ndarray,
                        kernel:np.ndarray,
                        lower:np.ndarray,
                        upper:np.ndarray,
                        template: Path,
                        threshold:float,
                        generate_video:bool=False,
                        eps:int=2,
                        start_frame:int=0
                        ):

    templ_img = cv2.imread(template, 0)
    h_t, w_t = templ_img.shape
    if path_out.exists():
        shutil.rmtree(path_out)
    path_out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(path_video.as_posix())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = None
    if generate_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_path = path_out/f'{path_video.stem}_mhI_output.mp4'
        # you have to update the final frame width according to the number of concatenated outputs in frame_concat
        frame_width = 2 * w
        writer = cv2.VideoWriter(file_path.as_posix(), fourcc, fps, (frame_width, h))

    # save previous frame contours
    prev_coords = None
    for i, _ in tqdm(enumerate(generator(cap)), total=nframes):
        ret, frame = cap.read()
        if not ret:
            break
        # table mask application
        frame = cv2.bitwise_and(frame,  frame, mask=mask)
        frame_blur = cv2.GaussianBlur(src=frame, ksize=(11, 11), sigmaX=0)
        # convert  frame to hsv
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, lower, upper)  # table's mask
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        # create inv mask
        mask_inv =cv2.bitwise_not(mask_green)
        masked_img = cv2.bitwise_and(frame_blur, frame_blur, mask=mask_inv)  # masked image with inverted mask
        masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        _, thresh_frame = cv2.threshold(src=masked_img_gray,
                                        thresh=30, maxval=255,
                                        type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
                                        )
        thresh_frame = cv2.GaussianBlur(src=thresh_frame, ksize=(7, 7), sigmaX=0)
        res = cv2.matchTemplate(thresh_frame, templ_img,
                                   cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)
        boxes = []
        probs = []
        for (x, y) in zip(*loc[::-1]):
            boxes.append((x, y, x + w_t, y + h_t))
            prob = res[y, x]
            probs.append(prob)
        # filtering boxes
        curr_coords = non_max_suppression(np.array(boxes),
                                          probs=probs,
                                          overlapThresh=0.7
                                          )
        if prev_coords is None:
            prev_coords = curr_coords
        for (cx1, cy1, cx2, cy2) in curr_coords:
            # detection flag switches to 1 when two contour centers are closer that eps
            det_flag = 0
            # filter blob which color does not match the balls colors
            to_filter = filter_by_color(hsv, (cx1, cy1, cx2, cy2))
            if not to_filter:
                for (px1, py1, px2, py2) in prev_coords:
                    # calculating distance between blob centers
                    if (cx1 - px1) ** 2 + (cy1 - py1) ** 2 < eps:
                        det_flag = 1
                        break
                if det_flag == 0:
                    cv2.circle(frame, (int(cx1 + w_t/2) , int(cy1 + h_t/2)), 11, (0, 255, 0), 2)
        prev_coords = curr_coords

        th_rgb = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2RGB)
        frame_concat = cv2.hconcat([frame, th_rgb])
        if not generate_video:
            frame_name = path_out / f'{i + start_frame:06}.png'
            cv2.imwrite(frame_name.as_posix(), frame_concat)
        else:
            writer.write(frame)
    if writer:
        writer.release()


if __name__ == '__main__':
    VIDEO = 'data/snooker_pix_5min.mp4'
    FOLDER_OUT = 'output/templ_matching_output'
    MASK = 'data/mask_im.jpg'
    TEMPLATE = 'template/template_bw1.png'
    GENERATE_VIDEO = False # generate video or separate frames
    # distance between previos and current blob centers in pixels
    EPS = 20
    START_FRAME = 320
    # template matching threshold
    THRESHOLD = 0.4
    # # snooker green HSV range:
    LOWER = np.array([28, 68, 36])
    UPPER = np.array([74, 244, 131])

    path_video = Path(VIDEO)
    path_out = Path(FOLDER_OUT)
    mask = cv2.imread(MASK, 0)

    kernel = np.ones((7, 7), np.uint8)
    detect_moving_balls(path_video=path_video,
                        path_out=path_out,
                        mask=mask,
                        kernel=kernel,
                        lower=LOWER,
                        upper=UPPER,
                        template = TEMPLATE,
                        threshold=THRESHOLD,
                        generate_video=GENERATE_VIDEO,
                        eps=EPS,
                        start_frame=START_FRAME
                        )