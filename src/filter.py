import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def findConnectedComponent(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 50 

    #your answer image
    img2 = np.zeros((output.shape), dtype=np.float32)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def main():
    img = cv2.imread("D:/pan.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.medianBlur(img,5)
    #th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    img2,contours,hierarchy = cv2.findContours(th1, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('img', th1)
    #cv2.waitKey(0)
    
    img1 = findConnectedComponent(th1)
    
    #img2 = 255-img2
    #cv2.imshow("Filtered image", img2)
    #cv2.waitKey(0)

    #Kernal to dilate the image in horizontal direction
    kernel = np.ones((1,11),np.uint8)
    img1 = cv2.dilate(img1,kernel,iterations = 1)
    cv2.imwrite('temp_filter.jpg', img1)
    img2 = cv2.imread('temp_filter.jpg')
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    #ret,th1 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
    #cv2.imwrite('D:/panres.jpg', img2)
    #cv2.imshow('Temp', img2)
    #cv2.waitKey(0)
    #gray=cv2.cvtColor(img2,cv2.COLOR_bina)
    img2,contours,hierarchy = cv2.findContours(th1, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if(w>h):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Text detection', img)
    cv2.waitKey(0)
    os.remove('temp_filter.jpg')

    #pilImg = Image.open("D:/pan.jpg")
    ##pilImg.show()
    #text = pytesseract.image_to_string(pilImg)
    #print(text)


if __name__ == "__main__":
    main()