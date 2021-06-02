import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

wheel = cv2.imread('19.jpg')
gray_img = cv2.cvtColor(wheel, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#cv2.imshow("wheel",cimg)
#cv2.waitKey(0)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=20,minRadius=10,maxRadius=18)
circles = np.uint16(np.around(circles))
circles.shape
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(wheel,(i[0],i[1]),i[2],(0,255,0),6)
    # draw the center of the circle
    cv2.circle(wheel,(i[0],i[1]),2,(0,0,255),3)

y,x,_ = plt.hist(circles[0,:,2], 30, (0,30))
#plt.plot(x[1:]-.5,y)
plt.xlim(10, 18)
plt.ylim(0, 45)
plt.savefig('img\output.png')
plt.show()
cv2.imwrite("img\Hough.jpg", wheel)
cv2.imshow("HoughCirlces", wheel)
cv2.waitKey()
cv2.destroyAllWindows()
