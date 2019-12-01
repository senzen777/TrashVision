#!/usr/bin/env python3
import cv2
import numpy as np
import imutils

class Union_find:
    def __init__(self):
        self.equiv_table = [-1]

    # Add label to the equivalence table
    def equiv_add(self, i):
        self.equiv_table.append(i)

    # Find the root label of a label
    def find(self, i):
        if i != self.equiv_table[int(i)]:
            i = self.find(self.equiv_table[int(i)])
        return i

    # Assign two labels to the same root label
    def union(self, i, j):
        i = self.find(self.equiv_table[int(i)])
        j = self.find(self.equiv_table[int(j)])
        if i != j:
            if i <= j:
                self.equiv_table[j] = self.equiv_table[i]
            else:
                self.equiv_table[i] = self.equiv_table[j]


def binarize(image, thresh_val):
    """Convert an image to a binary image using a threshold value.

    Args:
    - image: input image to convert
    - thresh_val: threshold value

    Return:
    - binary_image: image composed only of 0s and 255s
    """
    gray_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, binary_image = cv2.threshold(gray_image,thresh_val,255,cv2.THRESH_BINARY)
    return binary_image


def label(binary_image):
    """Label the different objects in a binary image.

    Args:
    - binary_image: image composed only of 0s and 255s

    Return:
    - labeled_image: image containing different objects colored in different shades of gray
    """
    height, width = binary_image.shape
    labeled_image = np.zeros((height, width))
    equiv_table = Union_find()
    label = 1
    for i in range(height):
        for j in range(width):
            # -1 if left or top is out of bounds
            left_label = -1
            top_label = -1

            # if not out of bounds and not 0 (background)
            if binary_image[i, j] > 0:
                # if left not out of bounds assign its value to left_label
                if(i - 1 >= 0) :
                    left_label = labeled_image[i-1, j]
                # if top not out of bounds assign its value to top_label
                if(j - 1 >= 0) :
                    top_label = labeled_image[i, j-1]

                # if left or top is exclusively not background and not out of bounds
                if (left_label > 0) != (top_label > 0):
                    # if it is left label that satisfies the condition
                    if left_label > 0:
                        labeled_image[i,j] = left_label
                    # if it is top label that satisfies the condition
                    else:
                        labeled_image[i,j] = top_label
                # if both left and top are not background and not out of bounds
                elif left_label > 0 and top_label > 0:
                    # if they are equal, pick one
                    if left_label == top_label:
                        labeled_image[i,j] = left_label
                    # if they are different, union them and assign the smaller value
                    else:
                        equiv_table.union(left_label, top_label)
                        if left_label < top_label:
                            labeled_image[i,j] = left_label
                        else:
                            labeled_image[i,j] = top_label
                # if pixel does not have valid neighbors, assign label, add to table, and increment label counter
                else:
                    labeled_image[i,j] = label
                    equiv_table.equiv_add(label)
                    label = label + 1

    # replace equivalent labels (second pass)
    for i in range(height):
        for j in range(width):
            if labeled_image[i,j] > 0:
                labeled_image[i,j] = equiv_table.find(labeled_image[i,j])

    # spread values out
    labeled_image = 255*labeled_image/np.amax(labeled_image)

    return labeled_image

def get_attribute(labeled_image):
    """Gets attributes of all the different labeled objects.

    Args:
    - labeled_image: image containing different objects colored in different shades of gray

    Return:
    - attribute_list: list of dictionaries containing x y position, orientation, and roundedness
    """
    height, width = labeled_image.shape
    attribute_list = []
    labels, area = np.unique(labeled_image, return_counts=True)

    # find all pixels corresponding to a label
    for i in range(len(labels)-1):
        y, x = np.where(labeled_image == labels[i+1])
        y = height - y - 1

        # find average x and y components corresponding to that label
        x_bar = float(sum(x)/area[i+1])
        y_bar = float(sum(y)/area[i+1])

        padding = 2

        # find min and max x and y values corresponding to that label
        x_min = int(np.amin(x)) - padding
        x_max = int(np.amax(x)) + padding
        y_min = int(height - np.amax(y) + 1) - padding
        y_max = int(height - np.amin(y) + 1) + padding

        a, b, c = [0, 0, 0]

        for i in range(len(x)):
            # make x and y values zero mean
            x[i] = x[i] - x_bar
            y[i] = y[i] - y_bar

            # obtain a, b, and c values incrementally
            a = a + x[i]**2
            b = b + 2*x[i]*y[i]
            c = c + y[i]**2

        # calculate corresponding theta
        theta = (np.arctan2(b, a-c))/2 # between -pi and pi

        # makes theta1 between 0 and pi if it isn't
        theta1 = float(theta)
        if theta1 < 0:
            theta1 = theta1 + np.pi

        # makes theta2 between 0 and pi if it isn't
        theta2 = float(theta1 + np.pi/2)
        if theta2 > np.pi:
            theta2 = theta2 - np.pi

        # calculates roundedness
        Emin = a*(np.sin(theta1))**2 - b*np.sin(theta1)*np.cos(theta1) + c*(np.cos(theta1))**2
        Emax = a*(np.sin(theta2))**2 - b*np.sin(theta2)*np.cos(theta2) + c*(np.cos(theta2))**2
        roundedness = float(Emin/Emax)

        # assigns all results to dictionary entry
        attributes = {
            "center": {
                "xbar": x_bar,
                "ybar": y_bar,
                },
            "bounds": {
                "xmin": x_min,
                "xmax": x_max,
                "ymin": y_min,
                "ymax": y_max,
                },
            "orientation": theta1,
            "roundedness": roundedness
        }

        # appends list of dictionaries
        attribute_list.append(attributes)

    return attribute_list

def orient(labeled_img, dst, attribute_list):
    objects = []
    for i in range(len(attribute_list)):
        dict = attribute_list[i]
        bounds = dict["bounds"]
        labeled_object = labeled_img[bounds["ymin"]: bounds["ymax"], bounds["xmin"]: bounds["xmax"]].copy()
        dst_obj = dst[bounds["ymin"]: bounds["ymax"], bounds["xmin"]: bounds["xmax"]].copy()

        unique, unique_counts = np.unique(labeled_object, return_counts=True)

        if len(unique) > 2:
            max_counts = 0
            for j in range(len(unique) - 1):
                if unique_counts[j+1] > max_counts:
                    important_label = unique[j+1]
                    max_counts = unique_counts[j+1]

            for j in range(dst_obj.shape[0]):
                for k in range(dst_obj.shape[1]):
                    if labeled_object[j,k] != important_label and labeled_object[j,k] != 0:
                        dst_obj[j,k] = 0

        object_rotated = imutils.rotate_bound(dst_obj, 180 - 180 * dict["orientation"] / np.pi)

        pad_height = 384
        pad_width = 512

        padded_object = np.zeros((pad_height, pad_width))
        padded_object[:object_rotated.shape[0], :object_rotated.shape[1]] = object_rotated.copy()

        objects.append(padded_object)

    return objects

def baseline_main():
    img = cv2.imread('test.png')
    binary_img =  binarize(img, 122)
    labeled_img = label(binary_img)
    labels = np.unique(labeled_img)

    attribute_list = get_attribute(labeled_img)

    cv2.imwrite('output/original.png', img)
    cv2.imwrite('output/binary.png', binary_img)
    cv2.imwrite('output/labeled.png', labeled_img)

    rectangle_image = cv2.imread('output/labeled.png', 1)
    for i in range (len(attribute_list)):
        dict = attribute_list[i]
        bounds = dict["bounds"]
        rectangle_img = cv2.rectangle(rectangle_image, ((bounds["xmin"]), (bounds["ymin"])), ((bounds["xmax"]), (bounds["ymax"])), (0,255,0), 2)

    cv2.imwrite('output/rectangles.png', rectangle_img)

    objects = orient(labeled_img, binary_img, attribute_list)
    for i in range(len(objects)):
        cv2.imwrite('output/object' + str(i) + '.png', objects[i])


if __name__ == '__main__':
    baseline_main()
