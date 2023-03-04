"""
THis script uses MHI (Motion History Image) in order to detect the balls motion.
Its advantage is that it is more sophisticated than a simple background subtraction because it indicates the motion by gray level intensity.
Since the mhi blobs corresponding to the latest frame have the highest gray level, they can be easily thresholded.
No need for ball coordinates tracking, since a resting ball is just a part of the background.
Its disadvantage,as you will see,
is that the background subtraction is less accurate in densely packed balls.
In this case the small  ball movements are undetected and the ball is detected after it disconnects from the joined blob
of the touching balls.

An output frame is created at the end of each processed frame.
It is a concatenation of four mid-step outputs:
the original frame after the mask application,
the original frame after the mask application with the green table subtracted,
the final thresholded mhi output, which is the input for the contour detection
the generated mhi image.

GENERATE_VIDEO - save the output frames in a video clip or save them separately.
"""

import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mhi import MotionHistory
from fiter_by_color import filter_by_color



def generator(cap):
    while cap.isOpened():
        yield


def generate_mhi(path_video: Path,
                 path_out: Path,
                 mask: np.ndarray,
                 kernel:np.ndarray,
                 lower: np.ndarray,
                 upper: np.ndarray,
                 mhi_history=10,
                 generate_video=False,
                 start_frame:int=0
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
        file_path = path_out / f'{path_video.stem}_mhI_output.mp4'
        # you have to update the final frame width according to the number of concatenated outputs in frame_concat
        frame_width = 4 * w
        writer = cv2.VideoWriter(file_path.as_posix(), fourcc, fps, (frame_width, h))

    motionHistory = MotionHistory(mhi_history, (h, w))
    subtractor = cv2.createBackgroundSubtractorKNN(history=mhi_history,
                                                   dist2Threshold=200,
                                                   detectShadows=False
                                                   )
    for i, _ in tqdm(enumerate(generator(cap)), total=nframes):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame_blur = cv2.GaussianBlur(src=frame, ksize=(11, 11), sigmaX=0)

        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        # green table mask definition
        mask_green = cv2.inRange(hsv, lower, upper)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_inv = cv2.bitwise_not(mask_green)  # mask inv
        # masked image after green table removal
        masked_img = cv2.bitwise_and(frame_blur, frame_blur, mask=mask_inv)
        # background subtraction
        fg_mask = subtractor.apply(masked_img)
        motionHistory.process(fg_mask)
        # mhi output
        mhi_img = motionHistory.scale_image()
        _, thresh_frame = cv2.threshold(src=mhi_img,
                                     thresh=(255//MHI_HISTORY)* MHI_HISTORY -1,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY
                                        )
        thresh_frame = cv2.GaussianBlur(src=thresh_frame, ksize=(7, 7), sigmaX=0)
        # thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel)
        thresh_frame = cv2.dilate(thresh_frame, (5, 5), iterations=1)
        contours, _ = cv2.findContours(image=thresh_frame,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE
                                       )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            aspect_r = w / h
            if cv2.contourArea(contour) > 250 and cv2.contourArea(contour) < 1600 and \
                    aspect_r > 0.7 and aspect_r < 10 / 7 and cv2.arcLength(contour, True) < 150 \
                    and cv2.arcLength(contour, True) > 60 and radius > 8 and radius < 20:
                # filter blob which color does not match the balls colors
                box = (x, y, x + w, y + h)
                to_filter = filter_by_color(hsv, box)
                if not to_filter:
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)

        th_rgb = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2RGB)
        mhi_rgb = cv2.cvtColor(mhi_img, cv2.COLOR_GRAY2RGB)
        frame_concat = cv2.hconcat([frame, masked_img, th_rgb, mhi_rgb])

        if not generate_video:
            frame_name = path_out / f'{i + start_frame:06}.png'
            cv2.imwrite(frame_name.as_posix(), frame_concat)
        else:
            writer.write(frame_concat)
    if writer:
        writer.release()


if __name__ == '__main__':
    VIDEO = 'data/snooker_pix_5min.mp4'
    FOLDER_OUT = 'output/mhi_output'
    MASK = 'data/mask_im.jpg'
    # generate video or separate frames
    GENERATE_VIDEO = False
    # number of frames for mhi generation
    MHI_HISTORY = 10
    # snooker green hsv range:
    LOWER = np.array([45, 70, 45])
    UPPER = np.array([75, 200, 122])

    path_video = Path(VIDEO)
    path_out = Path(FOLDER_OUT)
    mask = cv2.imread(MASK, 0)

    kernel = np.ones((7, 7), np.uint8)
    generate_mhi(path_video=path_video,
                 path_out=path_out,
                 mask=mask,
                 kernel=kernel,
                 lower=LOWER,
                 upper=UPPER,
                 mhi_history=MHI_HISTORY,
                 generate_video=GENERATE_VIDEO
                 )
