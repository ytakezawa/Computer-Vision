# yoshika takezawa
# 6/1/19
# kmeans segmentation oop implementation
# places to improve:
#   use hash sets for faster comparing, improve initial seeding,

import math
import random
import sys
import numpy as np
import cv2


class Cluster(object):
    def __init__(self, center):
        self. __center = center  # old center/mean
        self.__rolling_sum = [0, 0, 0]
        self.__numpoints = 0  # for rolling mean

    @property
    def center(self):
        return self.__center

    @property
    def rolling_sum(self):
        return self.__rolling_sum

    @property
    def numpoints(self):
        return self.__numpoints

    @center.setter
    def center(self, center):
        self.__center = center

    @rolling_sum.setter
    def rolling_sum(self, rolling_sum):
        self.__rolling_sum = rolling_sum

    @numpoints.setter
    def numpoints(self, numpoints):
        self.__numpoints = numpoints


class KMeans:
    def __init__(self, img, k):
        self. __img = img
        self. __k = k
        self.__my_clusters = []  # store for clusters
        self.__height = img.shape[0]
        self.__width = img.shape[1]

    @property
    def my_clusters(self):
        return self.__my_clusters

    @my_clusters.setter
    def my_clusters(self, my_clusters):
        self.__my_clusters = my_clusters

    def distance(self, b, g, r):
        smallest = 200000
        index = 0
        l = 0
        for cluster in self.__my_clusters:
            b1, g1, r1 = cluster.center
            # temp skips the sqrt b/c we are just comparing
            temp = (r-r1)**2+(g-g1)**2+(b-b1)**2
            if (temp < smallest):
                smallest = temp
                l = index
            index += 1
        return l

    def kmeans_main(self):
        # initializing random points
        temp = []       # making sure no color duplicates in initialization
        while(len(temp) < self.__k):
            x = random.randint(0, self.__width-1)
            y = random.randint(0, self.__height-1)
            t = [self.__img.item(y, x, 0), self.__img.item(
                y, x, 1), self.__img.item(y, x, 2)]
            if (t in temp):
                continue
            temp.append(t)
            self.__my_clusters.append(Cluster(t))
            print(t)

        is_equal = False
        # iterating through pixels until convergence
        while(not is_equal):
            # iterating through all pixels and finding the closest center
            for i in range(self.__width):
                for j in range(self.__height):
                    currb = self.__img.item(j, i, 0)
                    currg = self.__img.item(j, i, 1)
                    currr = self.__img.item(j, i, 2)
                    ind = self.distance(currb, currg, currr)
                    self.__my_clusters[ind].numpoints += 1

                    self.__my_clusters[ind].rolling_sum[0] += currb
                    self.__my_clusters[ind].rolling_sum[1] += currg
                    self.__my_clusters[ind].rolling_sum[2] += currr
            # checking if the mean and the center are equal
            string = ""
            is_equal = True
            for clust in self.__my_clusters:
                print(clust.numpoints)
                temp_mean = [int(clust.rolling_sum[0]/clust.numpoints), int(
                    clust.rolling_sum[1] / clust.numpoints), int(clust.rolling_sum[2]/clust.numpoints)]
                string += str(temp_mean) + " "
                if (clust.center != temp_mean):
                    is_equal = False
                # set rolling_mean as center and reset rolling and num
                clust.center = temp_mean
                clust.rolling_sum = [0, 0, 0]
                clust.numpoints = 0
            print(string)
        # printing image
        for x in range(self.__width):
            for y in range(self.__height):
                ib = self.__img.item(y, x, 0)
                ig = self.__img.item(y, x, 1)
                ir = self.__img.item(y, x, 2)
                i = self.distance(ib, ig, ir)
                self.__img.itemset((y, x, 0), self.__my_clusters[i].center[0])
                self.__img.itemset((y, x, 1), self.__my_clusters[i].center[1])
                self.__img.itemset((y, x, 2), self.__my_clusters[i].center[2])
        return self.__img


if __name__ == '__main__':

    # k means segamentation (part 1)
    wt_image = cv2.imread("white-tower.png", cv2.IMREAD_COLOR)
    km = KMeans(wt_image, 10)
    res1 = km.kmeans_main()
    cv2.imwrite('kmeans.png', res1)
