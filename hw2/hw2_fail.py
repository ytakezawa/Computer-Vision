# yoshika takezawa
# 5/1/19
# Computer Vision hw2
# I pledge my honor that i have abided by the stevens honor system
# THIS IS NOT THE FINAL VERSION. 
# kms -> week 6 pg 10
# slic -> week 6 pg 41

import math
import random
import sys
import numpy as np
import cv2

# given a pixel's bgr values, and the array of centroids,
# will output the index of the shortest distance


def bgr_distance(pix, bgrlist):
    # distance funct: sqrt((r-r1)^2+(g-g1)^2+(b-b1)^2)
    b, g, r = pix
    smallest = 200000
    for l in range(len(bgrlist)):
        b1, g1, r1 = bgrlist[l]
        # temp skips the sqrt b/c we are just comparing
        temp = (r-r1)**2+(g-g1)**2+(b-b1)**2
        if (temp < smallest):
            smallest = temp
            index = l
    return index


# k means segmentation takes image, randomly selects points as
# initial colors, processes image and colors until convergence


def kmeans(img, k=10):
    mean_lst = []
    cluster_lst = [[] for _ in range(k)]
    rows = img.shape[0]
    columns = img.shape[1]

    # getting first k unique random samples
    while(len(mean_lst) < k):
        x = random.randint(0, columns-1)
        y = random.randint(0, rows-1)
        mean = [img.item(y, x, 0), img.item(y, x, 1), img.item(y, x, 2)]
        if (mean in mean_lst):
            continue
        mean_lst.append(mean)
    # iterating through pixels until convergence
    while (1):
        # small_error = True
        temp_lst = []
        # iterating through pixels
        for i in range(columns):
            for j in range(rows):
                p = [img.item(j, i, 0), img.item(j, i, 1), img.item(j, i, 2)]
                ind = bgr_distance(p, mean_lst)
                cluster_lst[ind].append(p)
        # get mean for all clusters
        for c in range(len(cluster_lst)):
            temp = [(sum(l)//len(l)) for l in zip(*cluster_lst[c])]
            temp_lst.append(temp)
        # breaking condition
        if np.all(temp_lst == mean_lst):
            break
        else:
            mean_lst = temp_lst
            for cl in cluster_lst:
                cl.clear()
        print(mean_lst)
    # setting each pixel to its respective avg cluster color
    for n in range(columns):
        for m in range(rows):
            curbgr = [img.item(m, n, 0), img.item(m, n, 1), img.item(m, n, 2)]
            i = bgr_distance(curbgr, mean_lst)
            img.itemset((m, n, 0), mean_lst[i][0])
            img.itemset((m, n, 1), mean_lst[i][1])
            img.itemset((m, n, 2), mean_lst[i][2])
    return img

# given an image, square filter matrix and the length,
# outputs the filtered image


def filter(img, fm, len, channel):
    height, width = img.shape
    # create black image
    res = np.zeros((height, width), dtype=img.dtype)
    off = len//2
    for j in range(off, height-off):
        for i in range(off, width-off):
            fval = block_sum(img, fm, len, j, i)
            res.itemset((j, i), fval)
    return res

# using the filtermatrix's weight distribution, adds all the values


def block_sum(img, fm, len, y, x):
    total = 0.0
    off = len//2
    # iterating through each matrix element
    for j in range(len):
        for i in range(len):
            curx = x + i - off
            cury = y + j - off
            total += img.item(cury, curx) * fm[j][i]
    return total


def eudist(pix, ck):
    y, x, b, g, r = pix
    y1, x1, b1, g1, r1 = ck
    dist = ((x-x1)/2)**2 + ((y-y1)/2)**2 + (r-r1)**2+(g-g1)**2+(b-b1)**2
    return dist


def kmeans_5d(img, cent, S):
    rows = img.shape[0]
    columns = img.shape[1]
    pixel_lb_dst = [[[-1, float("inf")] for _ in range(columns)]
                    for _ in range(rows)]
    cluster_pix = [[] for _ in range(len(cent))]
    new_cent = [[] for _ in range(len(cent))]
    iter = 0
    while (1):
        iter += 1
        print(cent)
        # for each cluster center
        for k in range(len(cent)):
            # for each pixel
            for y in range(cent[k][0]-(2*S), cent[k][0] + (2*S)):
                for x in range(cent[k][1]-(2*S), cent[k][1] + (2*S)):
                    if (y < 0) | (x<0) | (y>rows-1) |(x > columns-1):
                        continue
                    # compute distance between pixel(i) and cluster center(k)
                    i = [y, x, img.item(y,x,0), img.item(y,x,0), img.item(y,x,0)]
                    dist = eudist(i, cent[k])
                    if (dist < pixel_lb_dst[y][x][1]):
                        old_k = pixel_lb_dst[y][x][0]
                        if old_k != -1:
                            cluster_pix[old_k].remove([y,x])
                        pixel_lb_dst[y][x][1] = dist
                        pixel_lb_dst[y][x][0] = k
                        cluster_pix[k].append([y,x])
        # compute new cluster centers
        for cc in cluster_pix:
            te = [(sum(l)//len(l)) for l in zip(*cc)]
            print(te)
            te.append(img.item(te[0], te[1], 0))
            te.append(img.item(te[0], te[1], 1))
            te.append(img.item(te[0], te[1], 2))
            new_cent.append(te)
        # compute residual error E
        E = 0
        for i in range(len(cent)):
            E += abs(cent[i][0]- new_cent[i][0])
            E += abs(cent[i][1]- new_cent[i][1])
            E += abs(cent[i][2]- new_cent[i][2])
            E += abs(cent[i][3]- new_cent[i][3])
            E += abs(cent[i][4]- new_cent[i][4])
        # breaking condition
        break
        # if (E <= 15) | (iter<10):
        #     break
        # else:
        #     cent = new_cent.copy()
        #     for j in range(len(cent)):
        #         cluster_pix[j].clear()
        #         new_cent[j].clear()
        #     E=0
    for y2 in range(rows):
        for x2 in range(columns):
            lb, ck = pixel_lb_dst[y2][x2]
            img.itemset((y2, x2, 0), cent[lb][2])
            img.itemset((y2, x2, 1), cent[lb][3])
            img.itemset((y2, x2, 2), cent[lb][4])
    return img


def SLIC(image, S=50):
    height = image.shape[0]
    width = image.shape[1]

    # initializing centroids in middle of 50x50 blocks
    centroids_coords = []
    for y in range(S//2, height, S):
        for x in range(S//2, width, S):
            centroids_coords.append([y, x,
                                     image.item(y, x, 0), image.item(y, x, 1), image.item(y, x, 2)])
    # compute bgr gradients and move centroids to the smallest gradient
    hsob = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
    vsob = [[1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]]
    # skipped sqrt in B/G/R_comb because will be squared later
    B_x = filter(image[:, :, 0], hsob, 3, 0)
    B_y = filter(image[:, :, 0], vsob, 3, 0)
    B_comb = np.power(B_x, 2.0) + np.power(B_y, 2.0)

    G_x = filter(image[:, :, 1], hsob, 3, 1)
    G_y = filter(image[:, :, 1], vsob, 3, 1)
    G_comb = np.power(G_x, 2.0) + np.power(G_y, 2.0)

    R_x = filter(image[:, :, 2], hsob, 3, 2)
    R_y = filter(image[:, :, 2], vsob, 3, 2)
    R_comb = np.power(R_x, 2.0) + np.power(R_y, 2.0)

    combined = B_comb + G_comb + R_comb
    # in the 3x3 windows surrounding center, finds smallest gradient
    for b in range(S//2, height, S):
        for a in range(S//2, width, S):
            index = (a//S) + ((b//S)*(width//S))
            small_temp = -1
            for t in range(b-1, b+2):
                for u in range(a-1, a+2):
                    cur_grad = combined.item(t, u)
                    if (cur_grad < small_temp) | (small_temp < 0):
                        small_temp = cur_grad
                        centroids_coords[index] = [t, u,
                                                   image.item(t, u, 0), image.item(
                                                       t, u, 1),
                                                   image.item(t, u, 2)]
    # use centroids as start for 5d kmeans
    kmeans5d = kmeans_5d(image, centroids_coords, S)

    # once the image stabilized, add black line b/w different color clusters
    return kmeans5d


if __name__ == '__main__':
    # k means segamnetation (part 1)

    # wt_image = cv2.imread("white-tower.png", cv2.IMREAD_COLOR)
    # seg_result = kmeans(wt_image, 10)
    # cv2.imwrite('kmeans.png', seg_result)

    # # SLIC (part 2)
    slic_image = cv2.imread("wt_slic.png", cv2.IMREAD_COLOR)
    slic_res = SLIC(slic_image, 150)
    cv2.imwrite('slic_fail.png', slic_res)
