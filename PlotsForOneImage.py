# import opencv and numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy.integrate import quad
from scipy.fft import fft, ifft






# global variable
H_low = 59
H_high = 106
S_low = 0
S_high = 255
V_low = 0
V_high = 255

# Choose which picture to process
picture = 780


# read source image

file = "DSC_0" + str(picture) + ".jpg"
img = cv2.imread(file)





path = "C:/Users/Tal/Desktop/pythonProject/processed"

#--------------------------------------------------------CROP----------------------------------------------------------



cv2.imshow("original", img)



cv2.waitKey(0)
cv2.destroyAllWindows()



# convert sourece image to HSC color mode
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#
hsv_low = np.array([H_low, S_low, V_low], np.uint8)
hsv_high = np.array([H_high, S_high, V_high], np.uint8)

# making mask for hsv range
mask = cv2.inRange(hsv, hsv_low, hsv_high)

# masking HSV value selected color becomes black
res = cv2.bitwise_and(img, img, mask=mask)

# show image
cv2.imshow('mask', mask)
cv2.imshow('res', res)

# destroys all window
cv2.destroyAllWindows()


#---------------------------------------------------------seperation----------------------------------------------------





# keep in mind that open CV loads images as BGR not RGB
image = res
cv2.waitKey(0)
cv2.destroyAllWindows()

## RESIZE IMAGE
# scale in percentage
scale = 60
newWidth = int(image.shape[1] * scale / 100)
newHeight = int(image.shape[0] * scale / 100)
newDimension = (newWidth, newHeight)

# resize image
resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the resized image
#cv2.imwrite("resizedParts.png", resizedImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
## CONVERT TO GRAYSCALE
# convert image to grayscale
grayImage=cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the transformed image
#cv2.imwrite("resizedPartsGray.png", grayImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# THRESHOLD
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#estimatedThreshold, thresholdImage=cv2.threshold(grayImage,50,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
estimatedThreshold, thresholdImage=cv2.threshold(grayImage,90,255,cv2.THRESH_BINARY)
# display converted image
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("resizedPartsThreshold.png", thresholdImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])


## DETERMINE CONTOURS AND FILTER THEM
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# make a copy of the resized image since we are going to draw contours on the resized image
resizedImageCopy = np.copy(resizedImage)

# draw all contours with setting the parameter to -1
# but if you use this function, you should comment the for loop below
# cv2.drawContours(resizedImageCopy,contours,-1,(0,0,255),2)
# filter contours
max_area = 0
for i, c in enumerate(contours):
    areaContour = cv2.contourArea(c)
    if areaContour > max_area:
        max_area = areaContour
        the_one = i






the_chosen = contours[the_one]
#_____________________________________AXIS____________________________________________________________________________
teta = []
teta_radian = []
radius_list = []
max_radius = 60
Test_teta = []
Test_radius = []
for i in range(0,(the_chosen.shape[0])):
    point = the_chosen[i,0]
    X_point = point[0]-267
    Y_point = point[1]-252

    Test_radi , Test_teti = cart2pol(X_point,Y_point)
    Test_radius.append(Test_radi)
    Test_teta.append(Test_teti)
    # on the y axis
    #resize radius_list.
    if X_point == 0:
        if abs(Y_point) < max_radius:
            radius = abs(Y_point)
            #radius_list.append(abs(Y_point))
        else:
            radius = max_radius
            #radius_list.append(max_radius)
        # resize radius.
            #vector yehida
            fixed_y = Y_point / abs(Y_point)
            point[0] = 267
            point[1] = (max_radius * fixed_y) + 252
        radius_list.append(radius)

        if Y_point < 0:
            teta.append(270)
            teta_radian.append(-np.pi/2)
        else:
            teta.append(90)
            teta_radian.append(np.pi/2)
        continue


    # on the x axis
    # resize radius.
    elif Y_point == 0:
        if abs(X_point) < max_radius:
            radius = abs(X_point)
            #radius_list.append(abs(X_point))
        else:
            radius = max_radius
            #radius_list.append(max_radius)
            fixed_x = X_point / abs(X_point)
            point[0] = (max_radius * fixed_x) + 267
            point[1] = 252
        radius_list.append(radius)

        if X_point < 0:
            teta.append(180)
            teta_radian.append(np.pi)
        else:
            teta.append(0)
            teta_radian.append(0)
        continue


    # X,Y !=0   check radius
    radius = np.sqrt(((X_point)**2)+((Y_point)**2))
    if radius > max_radius:
        fixed_x = X_point / radius
        fixed_y = Y_point / radius
        point[0] = (max_radius * fixed_x) + 267
        point[1] = (max_radius * fixed_y) + 252
        radius = max_radius

    radius_list.append(radius)
    Current_teta = np.arctan(Y_point/X_point) * 180 / np.pi
    if X_point < 0:
        if Y_point < 0:
            Current_teta = 180-abs(Current_teta)
        else:
            Current_teta = 270-abs(Current_teta)
    else:
        if Y_point < 0:
            Current_teta = 360-abs(Current_teta)


    teta.append(Test_teti)
    teta_radian.append(np.arctan(Y_point/X_point))


cv2.drawContours(resizedImageCopy, contours, the_one, (255, 10, 255), 4)




cv2.imshow('Image', resizedImageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(os.path.join(path, 'waka.jpg'), resizedImageCopy, [cv2.IMWRITE_PNG_COMPRESSION, 0])


 #********************************interpool-----------------------------------------

f = interpolate.interp1d(teta, radius_list)


period = 180
size_teta = len(teta_radian)
order = sorted(list(range(len(teta))), key=lambda i: teta[i])
soretrdTeta = [teta[i] for i in order]
sorterdraiu = [radius_list[i] for i in order]
y = radius_list[0:size_teta]

y = np.array(y)
teta_radian = np.array(teta_radian)





#--------------------------------------------------------------------------------------------------------------------
# Define domain
dx = 0.00001
L = np.pi





n=1024
w = np.exp(-2j*np.pi/n)
J,K = np.meshgrid(np.arange(n),np.arange(n))
DFT = np.power(w,J*K)
DFT = np.real(DFT)
#print(DFT)

plt.imshow(DFT)
plt.show()


# #********************************interpool-----------------------------------------
begin = 0
end = 360
jump = 45/4096
X_interp_Axis = np.arange(begin, end, jump)








f_sort = np.interp(X_interp_Axis,soretrdTeta,sorterdraiu)
plt.plot(X_interp_Axis, f_sort, color='r', label='interp')
plt.plot(soretrdTeta, sorterdraiu,'go', label='original')
plt.legend()
plt.show()
j = 0
while True:
    if f_sort[0] == max_radius:
        f_sort = np.roll(f_sort, 1)
    else:
        break


# # How many time points are needed i,e., Sampling Frequency

samplingFrequency = abs(soretrdTeta[1]-abs(soretrdTeta[2]))

# At what intervals time points are sampled

samplingInterval = 360/(len(sorterdraiu))

# Begin time period of the signals

beginTime = 0

# End time period of the signals

endTime = 360

# Frequency of the signals

signal1Frequency = 4

signal2Frequency = 7

# Time points

time = np.arange(beginTime, endTime, samplingInterval)

# Create two sine waves

amplitude1 = sorterdraiu



# Create subplot

figure, axis = plt.subplots(3, 1)

plt.subplots_adjust(hspace=1)

# Time domain representation for sine wave 1

axis[0].set_title('RADIUS(THETA)')

axis[0].plot(soretrdTeta, amplitude1)

axis[0].set_xlabel('Theta(deg)')

axis[0].set_ylabel('Radius')



amplitude = f_sort

# Time domain representation of the result

axis[1].set_title('interpolated function')

axis[1].plot(X_interp_Axis, amplitude)

axis[1].set_xlabel('Theta(deg)')

axis[1].set_ylabel('Radius')

# Frequency domain representation

fourierTransform = np.fft.fft(amplitude) / len(amplitude) # Normalize amplitude

fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency

fourierTransform[0] = 0

tpCount = len(amplitude)

values = np.arange(int(tpCount / 2))




N = len(amplitude) # Number of sample points
T = 1/len(amplitude) # sample frequency
FS = 1/jump

n_freq_bin = len(f_sort)
timestep = jump
freq_bin = np.fft.fftfreq(n_freq_bin, d=timestep)
frequencies = np.linspace(0.0, N/2, N//2+1)




max_amp =np.argmax(abs(fourierTransform))
ind = np.unravel_index(np.argmax(abs(fourierTransform), axis=None), fourierTransform.shape)
ind = ind[0]



# Frequency domain representation
x_ploty = ind
y_ploty = abs(fourierTransform[ind])

axis[2].set_title('Fourier transform depicting the wave number components')

axis[2].plot(frequencies[0:len(fourierTransform)], abs(fourierTransform), x_ploty,y_ploty,"o")

axis[2].set_xlabel('Wave number')

axis[2].set_ylabel('Amplitude')


text = "Wave number with largest amplitude: " + str(ind)
axis[2].annotate(text, xy=(x_ploty, y_ploty), xytext=(10, 1),arrowprops=dict(facecolor='black', shrink=0.05),color = "red")
plt.xlim(0, 30)



#-------------------------Freq Gauissian interpool---------------------------------------------------------------*
freq_rez = FS / len(fourierTransform)
part_0 = abs(fourierTransform[ind+1])/abs(fourierTransform[ind-1])
part_1 = (abs(fourierTransform[ind])**2)
part_2 = abs(fourierTransform[ind-1])*(abs(fourierTransform[ind+1]))
inter_bin = abs(fourierTransform[ind]) + (np.log(part_0))/(2*np.log(part_1/part_2))
interpool_freq = freq_rez * inter_bin


plt.show()

#
def cart2pol(x, y):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    Test_r,Test_theta = np.abs(z), np.angle(z,True)
    if Test_theta < 0:
        Test_theta += 360
    return Test_r,Test_theta