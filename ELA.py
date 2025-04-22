from PIL import Image
import numpy as np

class ErrorLevelAnalysis:
    def ELA(self, grayscaledImage, img_quality):
        originalImage = np.asarray(grayscaledImage, dtype=np.float64)
        compressedImage = Image.fromarray(grayscaledImage)
        compressedImage.save("compressed_image.jpg", quality=img_quality)
        compressedImage = np.array(Image.open("compressed_image.jpg"))
        
        elaImage = abs(originalImage - compressedImage)

        return elaImage
    
    def GetMeanStdPixels(self, img):
            # pixels = img
            if not isinstance(img, np.ndarray):
                pixels = list(img.getdata())
            else:
                pixels = img.flatten()

            mean = sum(pixels) / len(pixels)
            
            variance = sum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)
            std = variance ** 0.5
            
            return mean, std