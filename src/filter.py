import cv2
import numpy as np
import pytesseract
from PIL import Image

def findConnectedComponent(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 50 

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def main():
    img = cv2.imread("D:/pan.jpg",0)
    #img = cv2.medianBlur(img,5)
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('img', th1)
    cv2.waitKey(0)

    
    img2 = findConnectedComponent(th1)
    
    #img2 = 255-img2
    cv2.imshow("Filtered image", img2)

    kernel = np.ones((1,10),np.uint8)
    img2 = cv2.dilate(img2,kernel,iterations = 1)

    #cv2.imwrite('D:/panres.jpg', img2)
    cv2.imshow('Temp', img2)
    cv2.waitKey(0)
    pilImg = Image.open("D:/pan.jpg")
    #pilImg.show()
    text = pytesseract.image_to_string(pilImg)
    print(text)


if __name__ == "__main__":
    main()