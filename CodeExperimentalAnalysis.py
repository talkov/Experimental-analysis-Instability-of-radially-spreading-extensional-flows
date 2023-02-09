# import opencv and numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyefd import elliptic_fourier_descriptors
import os
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.interpolate import RegularGridInterpolator as rgi

# global variable
H_low = 59
H_high = 106
S_low = 0
S_high = 255
V_low = 0
V_high = 255

path = "C:/Users/Tal/Desktop/reserch/finale/finale countour"




# -----------------------------------------MAIN------------------------------------------- #


# MAIN function to run images through the pipeline #


def main():
    time_laps = {}
    time = 0
    Dominant_Wave_Numbers_List = []
    amp_matrix = []

    '''choose if you would like to show the image or not'''
    show = False

    '''specify the data set:'''
    Data_begins_at = 655
    Data_ends_at = 834
    # Data_begins_at = 730
    # Data_ends_at = 731
    lenData = Data_ends_at - Data_begins_at
    '''full data set:'''
    for picture in range(Data_begins_at, Data_ends_at+1):

        '''just to test image by image:'''
    #for picture in range(692, 693):
        #define option of axis calculation
        Axis_calc_option = 1

        #define wether to adjust the function by the begging of a finger
        roll_option = True
        # Build lists for interpolation decomposition
        theta_radian = []
        theta_list = []
        radius_list = []

        #define the maximum radius of trimmed fingers
        max_radius = 60


        # read source image
        file = "DSC_0" + str(picture) + ".jpg"
        img = cv2.imread(file)

        ## Call mask() function to get masked image from source image
        #masked_img =

        ## Call contour() function to get resized and masked image ready to fourier decomposition
        contour(mask(img), time, theta_radian, theta_list, radius_list, Axis_calc_option, max_radius,show)

        #intepolate function radius(theta)
        interpolated_function, sorted_theta, sorted_radius = Inter_func(radius_list, theta_list, roll_option, max_radius)

        #fourier decomposition
        Dominant_Wave_Number , z_axis_amplitude_list = fourier_decomposition(interpolated_function, sorted_theta, sorted_radius)

        Dominant_Wave_Numbers_List.append(Dominant_Wave_Number)

        max_amp = max(z_axis_amplitude_list[1:])
        normalized_amp = []
        for amplitude in z_axis_amplitude_list:
            normalized_amp.append(amplitude/max_amp)
        amp_matrix.append(normalized_amp[1:10])
        time_laps[time] = z_axis_amplitude_list

        # update time lapse
        time += 2


    #-------------------------b=tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
    amp_matrix = np.array(amp_matrix)
    print(amp_matrix.shape)
    #amp_matrix[:,0] = 0
    scale = 10000*abs(amp_matrix)
    amp_matrix = np.transpose(amp_matrix)
    amp_matrix = np.flipud(amp_matrix)
    print(amp_matrix.shape)
    plt.figsize = (6, 10)
    plt.imshow(abs(amp_matrix), extent=[0, 180, 1, 10], aspect='auto')
    plt.tight_layout()
    plt.xlabel("time(s)/2")
    plt.ylabel("Wave number")
    plt.colorbar()
    plt.show()



    time_jumps = np.arange(0,lenData+1)

    '''Display the plots'''
    Fig, ax = plt.subplots()
    for i, (mark, color) in enumerate(zip(
            ['s', 'o', 'D', 'v'], ['r', 'g', 'b', 'purple'])):
        ax.plot(i + 1, i + 1, color=color,marker=mark,markerfacecolor='None',markeredgecolor=color,label=i)
    ax.legend(numpoints=1)



# -----------------------------------------MASK------------------------------------------- #


## picture-> masked picture
def mask(img):
    # convert source image to HSV color mode
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #define mask array
    hsv_low = np.array([H_low, S_low, V_low], np.uint8)
    hsv_high = np.array([H_high, S_high, V_high], np.uint8)

    # making mask for hsv range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # masking HSV value selected color becomes black
    res = cv2.bitwise_and(img, img, mask=mask)


    # destroys all window
    cv2.destroyAllWindows()

    return res


# -----------------------------------------CONTOUR------------------------------------------- #

#masked image -> Image with contour
def contour(image, time, OG_teta_radian, OG_teta, OG_radius_list, Axis_calc_option, max_radius,show):

    # keep in mind that open CV loads images as BGR not RGB
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

    # convert image to grayscale
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

    # display converted image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # THRESHOLD
    estimatedThreshold, thresholdImage = cv2.threshold(grayImage, 90, 255, cv2.THRESH_BINARY)

    # display converted image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## DETERMINE CONTOURS AND FILTER THEM
    contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # make a copy of the resized image since we are going to draw contours on the resized image
    resizedImageCopy = np.copy(resizedImage)

    # filter contours
    max_area = 0
    the_one = 0
    for i, c in enumerate(contours):
        areaContour = cv2.contourArea(c)
        if areaContour > max_area:
            max_area = areaContour
            the_one = i
    name = str(time) + ".png"

    '''Axis calculation'''
    # CALL AXIS CALCULATION FUNCTION

    if Axis_calc_option == 1:
        '''option 1'''
        cart2pol(contours[the_one], OG_teta, OG_radius_list, max_radius)
    elif Axis_calc_option == 2:
        '''option 2'''
        Axis_calculation(contours[the_one], OG_teta_radian, OG_teta, OG_radius_list)

    # Draw Contours
    cv2.drawContours(resizedImageCopy, contours, the_one, (255, 10, 255), 4)

    '''Display image for tests'''
    if show:
        cv2.imshow('Image', resizedImageCopy)
        cv2.waitKey(0)

    # Save Image to file
    cv2.imwrite(os.path.join(path, name), resizedImageCopy, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return resizedImageCopy
# -----------------------------------------AXIS CALCULATION_OPTION_1------------------------------------------- #

'''-------------------------------------------------Option 1--------------------------------------------------'''

def cart2pol(the_chosen,theta_list,radius_list, max_radius):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''
    for i in range(0,(the_chosen.shape[0])):
        point = the_chosen[i,0]
        X_point = point[0]-267
        Y_point = point[1]-252

        '''In this option, we use the fundamental attributes of vectors in the complex plain (size and angle), 
        to extract the values of radius and theta that corresponds with the points on our contour '''
        z = X_point + Y_point * 1j
        radius, theta = np.abs(z), np.angle(z, True)
        if theta < 0:
            theta += 360
        '''Adjust the radius in the original contour'''
        if radius > max_radius:
            fixed_x = X_point / radius
            fixed_y = Y_point / radius
            point[0] = (max_radius * fixed_x) + 267
            point[1] = (max_radius * fixed_y) + 252
            radius = max_radius

        theta_list.append(theta)
        radius_list.append(radius)



'''-------------------------------------------------Option 2--------------------------------------------------'''
#appand the values to the lists Radius,teta in degrees, teta in radians.
#adjusts the radius to a normalized value
def Axis_calculation(the_chosen, teta_radian, teta, radius_list,max_radius):
    for i in range(0,(the_chosen.shape[0])):
        point = the_chosen[i,0]
        X_point = point[0]-267
        Y_point = point[1]-252

        # Y AXIS #
        if X_point == 0:
            if abs(Y_point) < max_radius:
                radius = abs(Y_point)
            else:
                radius = max_radius

                # resize radius by calculating unit vector
                fixed_y = Y_point / abs(Y_point)
                point[0] = 267
                point[1] = (max_radius * fixed_y) + 252
            radius_list.append(radius)

            if Y_point < 0:
                teta.append(-90)
                teta_radian.append(-np.pi/2)
            else:
                teta.append(90)
                teta_radian.append(np.pi/2)
            continue

        # X AXIS #
        elif Y_point == 0:
            if abs(X_point) < max_radius:
                radius = abs(X_point)
            else:
                radius = max_radius
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


        # If we got here X & Y are valid and we check the radius
        radius = np.sqrt((X_point ** 2) + (Y_point ** 2))
        if radius > max_radius:
            fixed_x = X_point / radius
            fixed_y = Y_point / radius
            point[0] = (max_radius * fixed_x) + 267
            point[1] = (max_radius * fixed_y) + 252
            radius = max_radius

        radius_list.append(radius)
        teta.append(np.arctan(Y_point/X_point) * 180 / np.pi)
        teta_radian.append(np.arctan(Y_point/X_point))


#------------------------------Interpolation function radius(theta)----------------------------------------------------
def Inter_func(radius_list, theta_list, roll_option, max_radius):

    '''sort theta list and then sort radius list accordingly'''
    order = sorted(list(range(len(theta_list))), key=lambda i: theta_list[i])
    sortedTeta = [theta_list[i] for i in order]
    sortedRadius = [radius_list[i] for i in order]

    '''Define sampling interval'''
    begin = 0
    end = 360
    jump = 45/4096
    X_interp_Axis = np.arange(begin, end, jump)

    '''Interpolation'''
    f_sort = np.interp(X_interp_Axis,sortedTeta,sortedRadius)

    '''optional move function to begin from beginning of the finger Phenomenon '''
    if roll_option:
        while True:
            if f_sort[0] == max_radius:
                f_sort = np.roll(f_sort, 1)
            else:
                break

    return f_sort, sortedTeta, sortedRadius
# -----------------------------------------FOURIER DECOMPOSITION------------------------------------------- #

#fourier decomposition and finding the  harmony of the largest amplitude

def fourier_decomposition(interpolation, sorted_theta, sorted_radius):

    '''Define sampling interval'''
    begin = 0
    end = 360
    jump = 45/4096
    X_interp_Axis = np.arange(begin, end, jump)


    fourierTransform = np.fft.fft(interpolation) / len(interpolation)  # Normalize amplitude

    fourierTransform = fourierTransform[range(int(len(interpolation) / 2))]  # Exclude sampling frequency
    Z_list = abs(fourierTransform)

    fourierTransform[0] = 0

    N = len(interpolation)
    WaveNumbers = np.linspace(0.0, N / 2, N // 2 + 1)


    ind = np.unravel_index(np.argmax(abs(fourierTransform), axis=None), fourierTransform.shape)
    ind = ind[0]


    wave_number = ind

    return wave_number , Z_list


if __name__ == "__main__":
    main()
