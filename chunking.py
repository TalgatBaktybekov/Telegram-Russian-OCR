import cv2 
import copy
import math
import numpy as np

def exposure_contrast(image):
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    exposed_image = np.uint8(np.clip(image_gray * 2.0, 0, 255))

    gamma = 1
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")

    return cv2.LUT(exposed_image, table)

# convert the image to black and white using threshold
def ThresholdImage(image, threshold):
    
    thresholded_image = np.zeros(image.shape)
    thresholded_image[image>=threshold] = 1
    
    return thresholded_image

# compute the otsu value for a threshold 
def ComputeOtsu(image, threshold):
    
    thresholded_image = ThresholdImage(image, threshold)

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
def FindThreshold(image):
    
    th_range = range(int(np.max(image))+1)
    otsus = [ComputeOtsu(image, threshold) for threshold in th_range]
    best_one = th_range[np.argmin(otsus)]
    
    return best_one        

# trim the image from both sides (bottom and top)/(right and left)
def Trim(image, vertical=False, rowFiltering=False):
    
    img = copy.deepcopy(image)

    if rowFiltering: 
        img = exposure_contrast(img)

    white_density = FindThreshold(img)/255

    if not rowFiltering:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    else:
        img = img/255

    img = ThresholdImage(img, white_density)
    
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

# Remove meaningless horizontal lines that are too close to each other, not enclosing anything (leave just one of them)
def HorFilter(arr):

    # arr - histogram of horizontal density of black pixels of the image

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

# Choose vertical lines from the given list, which split the rows the best, suitable to pass to the model
def VerFilter(row, ver_lines):

    width = row.shape[1]
    ver_lines.append(width)

    new_ver_lines = [ver_lines[0]]
    for i in range(1, len(ver_lines)):
        if row[:, new_ver_lines[-1]:ver_lines[min(i+1, len(ver_lines)-1)], :].shape[1] > 100:
            new_ver_lines.append(ver_lines[i])

    new_ver_lines.append(width)

    ver_lines = new_ver_lines

    # splitting the row into chunks of the best size to pass to the model
    n_splits = math.ceil((width - 100)/256) # calculating the number of splits 
    step = max(1, int(len(ver_lines)/n_splits)) # calculating the number of lines to skip, to keep every step-th line 
    
    # dummy head to avoid errors when accesing preceding elements in the list
    chunks = ['']

    # filtering the lines
    for i in range(0, len(ver_lines)-1, step):
        chunks.append(row[:, ver_lines[i]:ver_lines[min(i+step, len(ver_lines)-1)], :])

    return chunks[1:]

# Building histogram of an image to get the density of black pixels on defined axis
def Histogram(image_raw, vertical=False):
        
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)/255
    
    white_density = FindThreshold(image_raw)/255

    image = ThresholdImage(image, white_density)
    ax = 1 - vertical
    hist = image.shape[ax] - np.sum(image,axis=ax,keepdims=True)
    hist = hist.squeeze()
    if not vertical:
        for i in range(hist.shape[0]):
            hist[i] = hist[i]/max(image.shape[ax]/7,(Trim(image_raw[i, :, :][np.newaxis, :, :], vertical=True).shape[1]+1))
    else:
        for i in range(hist.shape[0]):
            hist[i] = hist[i]/max(image.shape[ax]/7,(Trim(image_raw[:, i, :][:, np.newaxis, :], vertical=False).shape[1]+1))
    hist = np.flip(hist)
    if not vertical:
        clip = 1 if vertical else 0.6
        hist = np.where(hist > clip, clip, hist)

        nonzero = hist[np.nonzero(hist)]
        average = np.mean(nonzero)

        hist = np.where(hist < average, 0, hist)
    
    return hist

# Based on the density of black pixels provided by histogram, segment it to rows/words
def LineSegmentation(hist, image_raw, vertical=False):
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

    if not vertical and len(lines) > 2:

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
def DrawLines(image_raw, lines, vertical=False):
    
    img = copy.deepcopy(image_raw)
    if vertical:
        for i in range(len(lines)):
            img = cv2.line(img, (lines[i], 0), (lines[i], image_raw.shape[0]), (255,0,0), 1)
    else:
        for i in range(len(lines)):
            img = cv2.line(img, (0, lines[i]), (image_raw.shape[1], lines[i]), (255,0,0), 1)
    return img

# Process the image and segment it into rows/words, get the image and pixels of lines 
def Process(image_raw, vertical=False):
    
    hist = Histogram(image_raw, vertical=vertical)
    
    if not vertical:
        hist = HorFilter(hist)
    
    lines = LineSegmentation(hist, image_raw, vertical=vertical)
    
    segmented_image = DrawLines(image_raw, lines, vertical=vertical)
    
    return segmented_image, lines

#
def ChunkImage(image_raw):

    # trim the image vertically and horizontally
    image_raw = Trim(Trim(image_raw, vertical=False), vertical=True)

    # get the pixels of lines for text rows
    _, hor_lines = Process(image_raw)
    
    # List of rows split into chunks(words)
    chunked_rows = []

    for i in range(len(hor_lines)-1):

        row = image_raw[hor_lines[i]:hor_lines[i+1]]
        row = Trim(Trim(row, vertical=False), vertical=True, rowFiltering=True)

        _, ver_lines = Process(row, vertical=True)
        
        # split the row into suitable chunks
        chunks = VerFilter(row, ver_lines)

        # add the chunked row to the list
        chunked_rows.append(chunks)

    return chunked_rows
