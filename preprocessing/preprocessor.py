import cv2
import numpy as np

class Preprocessor():
    """
    Provides the preprocess()-method to preprocess an image according to the configuration/options with which the Preprocessor was initialized.
    For an example see preprocessing_example.ipynb.
    Inspired by https://nanonets.com/blog/ocr-with-tesseract/.
    """

    def __init__(self, 
                 grayscale=False, 
                 noise=(False, 5, "median"), 
                 threshold=(False, 0), 
                 adapt_threshold=(False, 11, 2), 
                 dilation=False, 
                 erosion=False, 
                 opening=False, 
                 closing=False, 
                 canny=(False, 100, 200), 
                 deskew=False):
        (self.grayscale, 
        self.noise, 
        self.threshold, 
        self.adapt_threshold, 
        self.dilation, 
        self.erosion, 
        self.opening, 
        self.closing, 
        self.canny, 
        self.deskew) =  grayscale, noise, threshold, adapt_threshold, dilation, erosion, opening, closing, canny, deskew
        self.name = f"Preprocessor({grayscale}-{noise}-{threshold}-{adapt_threshold}-{dilation}-{erosion}-{opening}-{closing}-{canny}-{deskew})"


    def preprocess(self, img):
        if self.grayscale: img = self._to_grayscale(img)
        if self.noise[0]: img = self._remove_noise(img, strength=self.noise[1], type=self.noise[2])
        if self.threshold[0]: img = self._apply_threshold(img, thresh=self.threshold[0])
        if self.adapt_threshold[0]: img = self._apply_adaptive_threshold(img, blockSize=self.adapt_threshold[1], C=self.adapt_threshold[2])
        if self.dilation: img = self._dilate(img)
        if self.erosion: img = self._erode(img)
        if self.opening: img = self._apply_opening(img)
        if self.closing: img = self._apply_closing(img)
        if self.canny[0]: img = self._apply_canny(img, self.canny[1], self.canny[2])
        if self.deskew: img = self._correct_skew(img)

        return img


    def _to_grayscale(self, image):
        """
        Returns a grayscale version of the image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def _remove_noise(self, image, strength=5, type="median"):
        """
        Removes noise by applying blur (either median oder gaussian -> type).
        """
        if type=="median":
            return cv2.medianBlur(image, strength)
        elif type=="gaussian":
            return cv2.GaussianBlur(image, (5,5), 0)
 

    def _apply_threshold(self, image, thresh=0):
        """ 
        For every pixel, the threshold value is applied: If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value (here: 255).
        Should only be applied to grayscale images.
        Works best with gaussian blur applied before.
        """
        return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    def _apply_adaptive_threshold(self, image, blockSize=11, C=2):
        """
        Should only be applied to grayscale images.
        Calculates thresh value from the pixels in the blockSize*blockSize neighborhood and subtracts C.
        """
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)


    def _dilate(self, image):
        """
        Causes bright regions within an image to grow/expand (get bigger) and dark regions to shrink (get thinnner).
        Might help to   - removing noise
                        - isolate individual elements and join disparate elements
                        (- find intensity bumps or holes in an image).
        <--> erode
        """
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)


    def _erode(self, image):
        """
        Causes dark regions within an image to grow/expand (get bigger) and bright regions to shrink (get thinnner).
        Might help to   - removing noise
                        - isolate individual elements and join disparate elements
                        (- find intensity bumps or holes in an image).
        """
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations=1)


    def _apply_opening(self, image):
        """
        Erosion followed by dilation. Useful in removing noise.
        """
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


    def _apply_closing(self, image):
        """
        Dilation followed by erosion. Useful in closing small holes inside the foreground objects, or small black points on the object.
        """
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    def _apply_canny(self, image, thresh1=100, thresh2=200):
        """
        Returns the edge image (only detected edges are shown).
        """
        return cv2.Canny(image, thresh1, thresh2)


    def _correct_skew(self, image):
        """
        Rotates skewed images to correct rotation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        coords = np.column_stack(np.where(thresh < 255))
        angle = cv2.minAreaRect(coords)[-1]
        angle = angle - 90
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _remove_black_stains(self, image):
        """
        Removes black stains/spots from image.
        """
        # to be implemented
        return image
