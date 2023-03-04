import numpy as np

class MotionHistory:
    def __init__(self, mhi_duration, img_shape):
        self.mhi_duration = mhi_duration
        self.motion_history = np.zeros(img_shape, np.int32)

    def process(self, mask):
        # decrease values by 1 and verify they are between 0 and MHI_DURATION-1
        self.motion_history = np.clip(self.motion_history - 1, 0, self.mhi_duration - 1)
        # set value to MHI_DURATION for current iteration
        self.motion_history[mask != 0] = self.mhi_duration
        return self.motion_history

    def scale_image(self):
        scale_ratio = 255 // self.mhi_duration
        img_contrast = self.motion_history * scale_ratio
        return img_contrast.astype('uint8')