import cv2 
import copy
import math
import numpy as np

def threshold_image(image, threshold):
    
    thresholded_image = np.zeros(image.shape)
    thresholded_image[image>=threshold] = 1
    
    return thresholded_image

def compute_otsu(image, threshold):
    
    thresholded_image = threshold_image(image, threshold)

    pixels = image.size
    nonzeros = np.count_nonzero(thresholded_image)
    weight1 = nonzeros/pixels
    weight0 = 1 - weight1
    
    if weight1 == 0 or weight0 == 0:
        return np.inf
    var0 = np.var(image[thresholded_image==0]) if len(image[thresholded_image==0]) > 0 else 0
    var1 = np.var(image[thresholded_image==1]) if len(image[thresholded_image==1]) > 0 else 0
    
    return weight0 * var0 + weight1 * var1

def find_best_threshold(image):
    
    th_range = range(np.max(image)+1)
    otsus = [compute_otsu(image, threshold) for threshold in th_range]
    best_one = th_range[np.argmin(otsus)]
    
    return best_one        
    
def trimmer(image, vertical=False):
    
    img = copy.deepcopy(image)
    
    white_density = find_best_threshold(image)/255

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255

    _, img = cv2.threshold(img, white_density, 1, cv2.THRESH_BINARY)
    
    ax = 1 - vertical
    hist = img.shape[ax] - np.sum(img,axis=ax,keepdims=True)
    hist = hist.squeeze()
    pixels = len(hist) 
    pixel1 = 0
    pixel2 = pixels - 1
    while pixel1 < pixels and hist[pixel1] == 0:
        pixel1 += 1
    while pixel2 >= 0 and hist[pixel2] == 0:
        pixel2 -= 1
    return image[:, pixel1:pixel2+1, :] if vertical else image[pixel1:pixel2+1, :, :]


def parser(arr):
    flag = False
    length = len(arr)
    checkpoint = 0
    for i in range(length):

        if (arr[i] == 0) & flag:

            flag = False
            if i - checkpoint < min(math.ceil(length*0.01), 7):
                arr[checkpoint:i] = 0
            continue

        elif arr[i] == 0:
            continue

        elif not flag:
            flag = True
            checkpoint = i
            
    return arr


def histogram(image_raw, vertical=False):
        
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)/255
    
    white_density = find_best_threshold(image_raw)/255

    _, image = cv2.threshold(image, white_density, 1, cv2.THRESH_BINARY)
    ax = 1 - vertical
    hist = image.shape[ax] - np.sum(image,axis=ax,keepdims=True)
    hist = hist.squeeze()
    if not vertical:
        for i in range(hist.shape[0]):
            hist[i] = hist[i]/max(image.shape[ax]/7,(trimmer(image_raw[i, :, :][np.newaxis, :, :], vertical=True).shape[1]+1))
    else:
        for i in range(hist.shape[0]):
            hist[i] = hist[i]/max(image.shape[ax]/7,(trimmer(image_raw[:, i, :][:, np.newaxis, :], vertical=False).shape[1]+1))
    hist = np.flip(hist)
    if not vertical:
        clip = 1 if vertical else 0.6
        hist = np.where(hist > clip, clip, hist)

        nonzero = hist[np.nonzero(hist)]
        average = np.mean(nonzero)

        hist = np.where(hist < average, 0, hist)
    
    return hist

def line_segmentation(hist, vertical=False):

    gap = 0
    lines = []
    pixels = len(hist)
    pixel = 0
    coef = 0.01
    while (pixel < pixels):
        if hist[pixel] == 0:
            gap += 1
            pixel += 1
            continue
            
        elif gap >= max(2, int(pixels*coef)):
            lines.append(pixels - pixel + int(gap/2))
            
        gap = 0
        pixel += 1
        
    lines.append(0)
    lines.sort()
    
    return lines


def draw_lines(image_raw, lines, vertical=False):
    
    img = copy.deepcopy(image_raw)
    if vertical:
        for i in range(len(lines)):
            img = cv2.line(img, (lines[i], 0), (lines[i], image_raw.shape[0]), (255,0,0), 1)
    else:
        for i in range(len(lines)):
            img = cv2.line(img, (0, lines[i]), (image_raw.shape[1], lines[i]), (255,0,0), 1)
    return img


def process(image_raw, vertical=False):
    
    hist = histogram(image_raw, vertical=vertical)
    
    if not vertical:
        hist = parser(hist)
    
    lines = line_segmentation(hist, vertical=vertical)
    
    segmented_image = draw_lines(image_raw, lines, vertical=vertical)
    
    return segmented_image, lines

def ChunkImage(image_raw):

    _, hor_lines = process(image_raw)

    chunks = []

    for i in range(len(hor_lines)):

        if i == len(hor_lines) - 1:

            row = image_raw[hor_lines[i]:]
            row = trimmer(row, vertical=True)
            row = trimmer(row, vertical=False)

            _, ver_lines = process(row, vertical=True)

        else:

            row = image_raw[hor_lines[i]:hor_lines[i+1]]
            row = trimmer(row, vertical=True)
            row = trimmer(row, vertical=False)

            _, ver_lines = process(row, vertical=True)


        for i in range(0, len(ver_lines) - 1, 3):
            
            if i + 3 >= len(ver_lines)-1:
                chunks.append(row[:, ver_lines[i]:, :])
            else:
                chunks.append(row[:, ver_lines[i]:ver_lines[i+3], :])

    return chunks
