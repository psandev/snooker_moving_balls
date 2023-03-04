# Snooker Moving Balls

---

### Installation: Please, refer to requirements.txt



## I present three solutions:

## 1. Motion History Image (MHI)
    file: detection_mhi.py

    MHI's advantage is that it is more sophisticated than a simple background subtraction because it 
    indicates the motion by gray level intensity.
    Since the mhi blobs corresponding to the latest frame have the highest gray level, they can be 
    easily thresholded. No need for ball coordinates tracking, since a resting ball is just a part 
    of the background.
    Its disadvantage,as you will see, is that the background subtraction is less accurate in densely 
    packed balls. In this case the small ball movements are undetected and the ball is detected after
    it disconnects from the joined blob of the touching balls.
    
    An output frame is created at the end of each processed frame.
    It is a concatenation of four mid-step outputs:
    the original frame after the mask application,
    the original frame after the mask application with the green table subtracted,
    the final thresholded mhi output, which is the input for the contour detection
    the generated mhi image.

## 2. Background Removal by Color
    file: detection_color_background_removal.py

    In this script the green snooker table background is removed by using its hsv range.
    The remaining image is thresholded and the blobs detected.
    The blobs movement is detected by calculating the distance between the blobs in consecutive frames.
    A blob is plotted when the distance from all previous frame blobs is bigger than eps(paramemeter).
    As in the mhi script, a ball is detected when it separates from the other balls.

## 3. Template Matching (TM)
    file: detection_template_matching.py

    This script uses a matching template on the thresholded image after the snooker green table removal.
    The balls tracking is the same as in the color background removal script.
    THe idea behind using TM was to try to detect touching balls and this way to be 
    more sensitive to the small ball movements.
    TM did not perform well due to the different ball size, oclusions and touching balls. 
    Straigtening the image with homography will equalize the ball sizes, but the other issues remain.
    Also, if we want to draw bounding boxes on the original video, we will have to transform back and 
    the added inaquiracy from both transforms becomes too visible to tolerate.
    
## What else I tried:
    - Hough circles - it did not work well on the densely packed balls.
    - Color filter - I implemented it in order to get rid of blobs that are not balls by using the balls 
     color, but I couldn't get it to work properly. I couldn't get the color ranges right.


