"""
GENERATE_VIDEO(parameter) - save the output frames in a video clip or save them separately.
"""

import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
                        generate_video:bool = False,
                        eps:int = 2,
                        start_frame:int = 0
                        ):

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
        frame_width = 3 * w
        writer = cv2.VideoWriter(file_path.as_posix(), fourcc, fps, (frame_width, h))

    # save previous frame contours
    prev_contours = None
    for i, _ in tqdm(enumerate(generator(cap)), total=nframes):

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.bitwise_and(frame,  frame, mask=mask)
        frame_blur = cv2.GaussianBlur(src=frame, ksize=(11, 11), sigmaX=0)

        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)  # convert to hsv
        mask_green = cv2.inRange(hsv, lower, upper)  # table's mask
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_inv =cv2.bitwise_not(mask_green)  # mask inv
        masked_img = cv2.bitwise_and(frame_blur, frame_blur, mask=mask_inv)  # masked image with inverted mask
        masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        thresh_frame = cv2.threshold(src=masked_img_gray, thresh=30, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # thresh_frame = cv2.GaussianBlur(src=thresh_frame, ksize=(7, 7), sigmaX=0)
        # thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel)
        # thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)
        thresh_frame = cv2.erode(thresh_frame, (5, 5), iterations=1)
        thresh_frame = cv2.dilate(thresh_frame, (5, 5), iterations=1)
        curr_contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if prev_contours is None:
            prev_contours = curr_contours
        for c_cont in curr_contours:
            (cx, cy), cradius = cv2.minEnclosingCircle(c_cont)
            # detection flag switches to 1 when two contour centers are closer that eps
            det_flag = 0
            box = (cx - cradius, cy - cradius, cx + cradius, cy + cradius)
            # filter blob which color does not match the balls colors
            to_filter = filter_by_color(hsv, box)
            if not to_filter:
                if cradius > 4 and cradius < 16:
                    for p_cont in prev_contours:
                        (px, py) , pradius = cv2.minEnclosingCircle(p_cont)
                        # calculating distance between blob centers
                        if (cx - px) ** 2 + (cy - py) ** 2 < eps:
                            det_flag = 1
                            break
                    if det_flag == 0:
                        cv2.circle(frame, (int(cx), int(cy)), int(cradius), (0, 255, 0), 2)

        prev_contours = curr_contours
        th_rgb = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2RGB)
        frame_concat = cv2.hconcat([frame, masked_img, th_rgb])

        if not generate_video:
            frame_name = path_out / f'{i + start_frame:06}.png'
            cv2.imwrite(frame_name.as_posix(), frame_concat)
        else:
            writer.write(frame_concat)

    if writer:
        writer.release()

if __name__ == '__main__':
    VIDEO = 'data/snooker_pix_5min.mp4'
    FOLDER_OUT = 'output/bg_removal_output'
    MASK = 'data/mask_im.jpg'
    # generate video or separate frames
    GENERATE_VIDEO = False
    # distance between previos and current blob centers in pixels
    EPS = 2
    START_FRAME = 320
    # snooker green HSV range:
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
                        generate_video=GENERATE_VIDEO,
                        eps=EPS,
                        start_frame=START_FRAME
                        )