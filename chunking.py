import cv2 
import copy
import math
import numpy as np

# convert the image to black and white using threshold
def threshold_image(image, threshold):
    
    thresholded_image = np.zeros(image.shape)
    thresholded_image[image>=threshold] = 1
    
    return thresholded_image

# compute the otsu value for a threshold 
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

# compute all of the otsu values for and image and find the best threshold
def find_best_threshold(image):
    
    th_range = range(np.max(image)+1)
    otsus = [compute_otsu(image, threshold) for threshold in th_range]
    best_one = th_range[np.argmin(otsus)]
    
    return best_one        

# trim the image from both sides (bottom and top)/(right and left)
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

# Remove meaningless lines that are too close to each other, not enclosing anything (leave just one of them)
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

# Building histogram of an image to get the density of black pixels on defined axis
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

# Based on the density of black pixels provided by histogram, segment it to rows/words
def line_segmentation(hist, image_raw, vertical=False):
    # the logic of the segmenting is to find white gaps big enough and take the middle of the gap as the pixel for split
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

    if not vertical:
        if len(lines) > 2:

            # need to filter the lines first, try to leave only ones enclosing text rows
            
            # calculate the distance between pixels of lines
            differences = [lines[i+1] - lines[i] for i in range(len(lines) - 1)]
            
            # average distance 
            avg_diff = sum(differences) / len(differences)
            
            # half of the average
            half_avg_diff = avg_diff / 2

            # hor_lines[0] is 0
            result = [lines[0]]  

            for i in range(1, len(lines)):

                # if the last two lines are too close to each other, remove them (this will extend the previous row until the end of the image)
                if i == len(lines) - 2 and abs(differences[i]) <= half_avg_diff:
                    result.append(image_raw.shape[0])
                    break

                # if the distance between the line and the previous one is long enough, keep the line 
                elif i == len(lines) - 1 or abs(differences[i-1]) >= half_avg_diff:
                    result.append(lines[i])

        lines = result
    
    return lines

# Draw lines if want to see the segmentation of the image
def draw_lines(image_raw, lines, vertical=False):
    
    img = copy.deepcopy(image_raw)
    if vertical:
        for i in range(len(lines)):
            img = cv2.line(img, (lines[i], 0), (lines[i], image_raw.shape[0]), (255,0,0), 1)
    else:
        for i in range(len(lines)):
            img = cv2.line(img, (0, lines[i]), (image_raw.shape[1], lines[i]), (255,0,0), 1)
    return img

# Process the image and segment it into rows/words, get the image and pixels of lines 
def process(image_raw, vertical=False):
    
    hist = histogram(image_raw, vertical=vertical)
    
    if not vertical:
        hist = parser(hist)
    
    lines = line_segmentation(hist, image_raw, vertical=vertical)
    
    segmented_image = draw_lines(image_raw, lines, vertical=vertical)
    
    return segmented_image, lines

#
def ChunkImage(image_raw):

    # trim the image vertically and horizontally
    image_raw = trimmer(trimmer(image_raw, vertical=False), vertical=True)

    # get the pixels of lines for text rows
    _, hor_lines = process(image_raw)

    # List of rows split into chunks(words)
    chunked_rows = []

    if image_raw.shape[1] <= 256:
        return chunked_rows.append(image_raw)

    for i in range(len(hor_lines)-1):
        
        # dummy head to avoid errors when accesing preceding elements in the list
        chunks = ['']

        row = image_raw[hor_lines[i]:hor_lines[i+1]]
        row = trimmer(trimmer(row, vertical=False), vertical=True)

        width = row.shape[1]

        _, ver_lines = process(row, vertical=True)

        # splitting the row into chunks of the best size to pass to the model
        n_splits = math.ceil((width - 100)/256) # calculating the number of splits 
        step = max(1, int(len(ver_lines)/n_splits)) # calculating the number of lines to skip, to keep every step-th line 
        
        ver_lines.append(width)

        # filtering the lines
        for i in range(0, len(ver_lines)-1, step):

            if i > 0 and row[:, ver_lines[i]:ver_lines[min(i+step, len(ver_lines)-1)], :].shape[1] <= 100:
                chunks[-1] = np.concatenate((chunks[-1], row[:, ver_lines[i]:ver_lines[min(i+step, len(ver_lines)-1)], :]), axis=1)
            else:
                chunks.append(row[:, ver_lines[i]:ver_lines[min(i+step, len(ver_lines)-1)], :])

        # add the chunked row to the list
        chunked_rows.append(chunks[1:])

    return chunked_rows
