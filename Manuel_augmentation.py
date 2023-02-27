# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:31:21 2023

@author: Christopher Reichel
"""

import os
import math
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from PIL import Image
import random



DATAPATH = os.path.join('C:/Users/chris/Desktop/Studium/Masterarbeit/bs-data/')

# %% load input data

# load the training dataset
pickle_in = open(os.path.join(DATAPATH, 'X_og.pickle'), 'rb')
X = pickle.load(pickle_in)

# load the categories for the training data set
pickle_in = open(os.path.join(DATAPATH, 'y_og.pickle'), 'rb')
y = pickle.load(pickle_in)


fillvalue = -999
X[X == fillvalue] = 0

# %% horizontal and vertical cut


cut_data_all = np.zeros((1155,224,180))
i = 0
for data in X:
    
    #horizontal cut
    cut_data_all[i] = X[i, 0:224, 0:180,0]
    
    #vertical cut
    #comment out if needed
    #cut_data_all[i] = X[i, 0:180, 0:224,0]
    
    
    i = i + 1


# cut data is not 224x224 anymore, so we need to resize it

#  resize images to 224x224
    
basewidth = 224
hsize = 224
cut_image_all = []
cut_image_array_all= np.zeros((1155,224,224))

for i in range(1155):
    
    cut_image_all.append(Image.fromarray(cut_data_all[i])) #array in image umwandeln
     

    # resize image and save
    cut_image_all[i] = cut_image_all[i].resize((basewidth,hsize), Image.ANTIALIAS) #alternativ NEAREST
      


    #change image into array
    cut_image_all[i] = cut_image_all[i].getdata()

    cut_image_array_all[i] = np.reshape(np.array(cut_image_all[i]), (224,-1))
    
    
    





X_horizontal_shift = cut_image_array_all
X_horizontal_shift = np.array(cut_image_array_all).reshape(-1, 224, 224, 1)
y_horizontal_shift = y
    
    
# compare cut data and original data
    
#plot resized image
plt.figure()
cb = plt.imshow(cut_image_array_all[2,:,:], vmin = 0, vmax = 1)
plt.colorbar(cb)    

#plot cropped image
plt.figure()
cb = plt.imshow(cut_data_all[2,:,:], vmin = 0, vmax = 1)
plt.colorbar(cb)    

#plot original image
plt.figure()
cb = plt.imshow(X[2,:,:,0], vmin = 0, vmax = 1)
plt.colorbar(cb)    


# save data 
#change horizontal to vertical if needed


pickle_out = open(os.path.join(DATAPATH, 'X_horizontal_shift.pickle'), 'wb')
pickle.dump(X_horizontal_shift, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATAPATH, 'y_horizontal.pickle'), 'wb')
pickle.dump(y_horizontal_shift, pickle_out)
pickle_out.close()
    


# %% horizontal and vertical flip


datagen_flip_hori = ImageDataGenerator(
        horizontal_flip=True, 
        fill_mode='constant')   

datagen_flip_verti = ImageDataGenerator(
        vertical_flip=True, 
        fill_mode='constant')  

# split dataset into the three categories, because ImageDataGenerator  randomize all images

i=0
# 0 = open
# 1 = closed
# 2 = no mcc

x_0 = []                   
y_0 = []

x_1 = []                    
y_1 = []

x_2 = []                   
y_2 = []
for category in y:                            #Datensatz von 1155 Bildern wieder in 3 Datensätze mit der jeweiligen Kategorie aufsplitten
    if category == 0 :
        x_0.append(X[i])
        y_0.append(category)
    
    elif category == 1 :
        x_1.append(X[i])
        y_1.append(category)
        
    elif category == 2 :
        x_2.append(X[i])
        y_2.append(category)
        
    i = i + 1
        
x_open = np.array(x_0).reshape(-1, 224, 224, 1)    #von liste in array transferieren
x_closed = np.array(x_1).reshape(-1, 224, 224, 1)  
x_nomcc = np.array(x_2).reshape(-1, 224, 224, 1)  




# horizontal flip

aug_open = []    
i = 0
for batch in datagen_flip_hori.flow(x_open, batch_size=1,
                          #save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/fliptest/open', 
                          #save_prefix='aug', 
                          #save_format='png'
                          ):  
                        
    aug_open.append(batch)

    i += 1
    if i > 384:
        break  

#uncomment next lines if vertical flip is needed, also change horizontal names to vertical names if needed
        
# for batch in datagen_flip_verti.flow(x_open, batch_size=1,
#                           #save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/fliptest/open', 
#                           #save_prefix='aug', 
#                           #save_format='png'
#                           ):  
                        
#     aug_open.append(batch)

#     i += 1
#     if i > 384:
#         break  

y_oo = []
for i in range(385): 
    y_oo.append(0)
    
    
aug_list_open = []
for scene in range(0, len(y_oo)):
    aug_list_open.append([aug_open[scene], y_oo[scene]])



aug_closed = []    
i = 0
for batch in datagen_flip_hori.flow(x_closed, batch_size=1,
                          # save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/fliptest/closed', 
                          # save_prefix='aug', 
                          # save_format='png'
                          ):  
                        
    aug_closed.append(batch)

    i += 1
    if i > 384:
        break 



y_cc = []
for i in range(385): 
    y_cc.append(1)
    
    
aug_list_closed = []
for scene in range(0, len(y_cc)):
    aug_list_closed.append([aug_closed[scene], y_cc[scene]])


aug_nomcc = []    
i = 0
for batch in datagen_flip_hori.flow(x_nomcc, batch_size=1,
                          # save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/fliptest/nomcc', 
                          # save_prefix='aug', 
                          # save_format='png'
                          ):  
                        
    aug_nomcc.append(batch)
    i += 1
    if i > 384:
        break



y_nm = []
for i in range(385): 
    y_nm.append(2)
    
    
aug_list_nomcc = []
for scene in range(0, len(y_nm)):
    aug_list_nomcc.append([aug_nomcc[scene], y_nm[scene]])
    
    
    
# sum all aug_list to one
    
    
aug_biglist = []

aug_biglist = aug_list_open + aug_list_closed + aug_list_nomcc

# random shuffel training data   
    
random.shuffle(aug_biglist)
    

for data in aug_biglist:
     #print(sample[1])

    X_hori_flip = []
    y_hori_flip = []

    for data, label in aug_biglist:
        X_hori_flip.append(data)
        y_hori_flip.append(label)


X_hori_flip = np.array(X_hori_flip).reshape(-1, 224, 224, 1)




pickle_out = open(os.path.join(DATAPATH, 'X_hori_flip.pickle'), 'wb')
pickle.dump(X_hori_flip, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATAPATH, 'y_hori_flip.pickle'), 'wb')
pickle.dump(y_hori_flip, pickle_out)
pickle_out.close()
    



# %% zoom in by a factor of 30%

datagen_zoom = ImageDataGenerator(
        zoom_range=(-0.7,-0.7), 
        fill_mode='constant')





i=0
x_0 = []                    
y_0 = []

x_1 = []                    
y_1 = []

x_2 = []                    
y_2 = []

for category in y:                            
    if category == 0 :
        x_0.append(X[i])
        y_0.append(category)
    
    elif category == 1 :
        x_1.append(X[i])
        y_1.append(category)
        
    elif category == 2 :
        x_2.append(X[i])
        y_2.append(category)
        
    i = i + 1
        
x_open = np.array(x_0).reshape(-1, 224, 224, 1)    
x_closed = np.array(x_1).reshape(-1, 224, 224, 1)  
x_nomcc = np.array(x_2).reshape(-1, 224, 224, 1)  

aug_open = []    
i = 0
for batch in datagen_zoom.flow(x_open, batch_size=1,
                          #save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/zoomtest/open', 
                          #save_prefix='aug', 
                          #save_format='png'
                          ):  
                        
    aug_open.append(batch)

    i += 1
    if i > 384:
        break  #


y_oo = []
for i in range(385): 
    y_oo.append(0)
    
    
aug_list_open = []
for scene in range(0, len(y_oo)):
    aug_list_open.append([aug_open[scene], y_oo[scene]])


aug_closed = []    
i = 0
for batch in datagen_zoom.flow(x_closed, batch_size=1,
                          #save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/zoomtest/closed', 
                          #save_prefix='aug', 
                          #save_format='png'
                          ):  
                        
    aug_closed.append(batch)

    i += 1
    if i > 384:
        break  



y_cc = []
for i in range(385): 
    y_cc.append(1)
    
    
aug_list_closed = []
for scene in range(0, len(y_cc)):
    aug_list_closed.append([aug_closed[scene], y_cc[scene]])



aug_nomcc = []    
i = 0
for batch in datagen_zoom.flow(x_nomcc, batch_size=1,
                          #save_to_dir='C:/Users/chris/Desktop/Studium/dataaug/zoomtest/nomcc', 
                          #save_prefix='aug', 
                          #save_format='png'
                          ):  
                        
    aug_nomcc.append(batch)
    i += 1
    if i > 384:
        break 



y_nm = []
for i in range(385): 
    y_nm.append(2)
    
    
aug_list_nomcc = []
for scene in range(0, len(y_nm)):
    aug_list_nomcc.append([aug_nomcc[scene], y_nm[scene]])


    
aug_biglist = []

aug_biglist = aug_list_open + aug_list_closed + aug_list_nomcc
    
      
    
random.shuffle(aug_biglist)


for data in aug_biglist:
     

    X_zoom = []
    y_zoom = []

    for data, label in aug_biglist:
        X_zoom.append(data)
        y_zoom.append(label)


X_zoom = np.array(X_zoom).reshape(-1, 224, 224, 1)


pickle_out = open(os.path.join(DATAPATH, 'X_zoom.pickle'), 'wb')
pickle.dump(X_zoom, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATAPATH, 'y_zoom.pickle'), 'wb')
pickle.dump(y_zoom, pickle_out)
pickle_out.close()
    
# %% add brightness to training data

X_brightness = X*0.8
y_brightness = y

pickle_out = open(os.path.join(DATAPATH, 'X_brightness.pickle'), 'wb')
pickle.dump(X_brightness, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATAPATH, 'y_brightness.pickle'), 'wb')
pickle.dump(y_brightness, pickle_out)
pickle_out.close()


# %%  rotate and crop



def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]



    """
    largest_rotated_rect function
    """



i=0
x_0 = []                    
y_0 = []

x_1 = []                    
y_1 = []

x_2 = []                    
y_2 = []

for category in y:                            
    if category == 0 :
        x_0.append(X[i])
        y_0.append(category)
    
    elif category == 1 :
        x_1.append(X[i])
        y_1.append(category)
        
    elif category == 2 :
        x_2.append(X[i])
        y_2.append(category)
        
    i = i + 1
        
x_open = np.array(x_0).reshape(-1, 224, 224)    
x_closed = np.array(x_1).reshape(-1, 224, 224)  
x_nomcc = np.array(x_2).reshape(-1, 224, 224)


# rot = 60 means the image will be rotated by 60 degree
# change the size of degree to get more augmented data
rot = 60

#open


image_height = 224
image_width = 224

image_orig_open = np.zeros((385,224,224))
image_rotated_open = np.zeros((385,305,305))
image_rotated_cropped_open = np.zeros((385,163,163))
image_rotated_cropped_open_resized = np.zeros((385,224,224))


for j in range(385):
    
   
        image_orig_open[j] = np.copy(x_open[j])
        image_rotated_open = rotate_image(x_open[j], rot)
        image_rotated_cropped_open[j] = crop_around_center(
            image_rotated_open,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(60)
            )
        )
    
    



        # set the base width of the result
        basewidth = 224
        img = Image.fromarray(image_rotated_cropped_open[j])

        # determining the height ratio
        wpercent = (basewidth/float(img.size[0]))
        hsize = 224
        
        # resize image and save
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)  


        #change image into array
        image_sequence = img.getdata()

        image_rotated_cropped_open_resized[j] = np.reshape(np.array(image_sequence), (224,-1))
        
        
plt.figure()
cb = plt.imshow(image_rotated_cropped_open_resized[0], vmin = 0, vmax = 1)
plt.colorbar(cb)


plt.figure()
cb = plt.imshow(image_rotated_cropped_open[0], vmin = 0, vmax = 1)
plt.colorbar(cb)        
        
        
plt.figure()
cb = plt.imshow(x_open[0], vmin = 0, vmax = 1)
plt.colorbar(cb)



#closed


image_height = 224
image_width = 224

image_orig_closed = np.zeros((385,224,224))
image_rotated_closed = np.zeros((385,305,305))
image_rotated_cropped_closed = np.zeros((385,163,163))
image_rotated_cropped_closed_resized = np.zeros((385,224,224))


for j in range(385):
    
   
        image_orig_closed[j] = np.copy(x_closed[j])
        image_rotated_closed = rotate_image(x_closed[j], rot)
        image_rotated_cropped_closed[j] = crop_around_center(
            image_rotated_closed,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(60)
            )
        )
    
    
        # set the base width of the result
        basewidth = 224
        img = Image.fromarray(image_rotated_cropped_closed[j])

        # determining the height ratio
        wpercent = (basewidth/float(img.size[0]))
        hsize = 224
        # resize image and save
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)

        #change image into array
        image_sequence = img.getdata()

        image_rotated_cropped_closed_resized[j] = np.reshape(np.array(image_sequence), (224,-1))
        
        
plt.figure()
cb = plt.imshow(image_rotated_cropped_closed_resized[0], vmin = 0, vmax = 1)
plt.colorbar(cb)


plt.figure()
cb = plt.imshow(image_rotated_cropped_closed[0], vmin = 0, vmax = 1)
plt.colorbar(cb)        
        
        
plt.figure()
cb = plt.imshow(x_closed[0], vmin = 0, vmax = 1)
plt.colorbar(cb)        


#nomcc


image_height = 224
image_width = 224

image_orig_nomcc = np.zeros((385,224,224))
image_rotated_nomcc = np.zeros((385,305,305))
image_rotated_cropped_nomcc = np.zeros((385,163,163))
image_rotated_cropped_nomcc_resized = np.zeros((385,224,224))



for j in range(385):
    
   
        image_orig_nomcc[j] = np.copy(x_nomcc[j])
        image_rotated_nomcc = rotate_image(x_nomcc[j], rot)
        image_rotated_cropped_nomcc[j] = crop_around_center(
            image_rotated_nomcc,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(60)
            )
        )
    
    



        # set the base width of the result
        basewidth = 224
        img = Image.fromarray(image_rotated_cropped_nomcc[j])

        # determining the height ratio
        wpercent = (basewidth/float(img.size[0]))
        hsize = 224
        # resize image and save
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)


        #change image into array
        image_sequence = img.getdata()

        image_rotated_cropped_nomcc_resized[j] = np.reshape(np.array(image_sequence), (224,-1))
        
        
plt.figure()
cb = plt.imshow(image_rotated_cropped_nomcc_resized[0], vmin = 0, vmax = 1)
plt.colorbar(cb)


plt.figure()
cb = plt.imshow(image_rotated_cropped_nomcc[0], vmin = 0, vmax = 1)
plt.colorbar(cb)        
        
        
plt.figure()
cb = plt.imshow(x_nomcc[0], vmin = 0, vmax = 1)
plt.colorbar(cb)                
        
# sum all images

image_rotated_cropped_all = np.zeros((1155,224,224))


image_rotated_cropped_all[0:385,:,:] = image_rotated_cropped_open_resized

image_rotated_cropped_all[385:770,:,:] = image_rotated_cropped_closed_resized

image_rotated_cropped_all[770:1155,:,:] = image_rotated_cropped_nomcc_resized

image_rotated_cropped_all = np.array(image_rotated_cropped_all).reshape(-1, 224, 224, 1)  


X_rotated_cropped = image_rotated_cropped_all

y_rotated_cropped = np.zeros(1155)


y_rotated_cropped[0:385] = 0

y_rotated_cropped[385:770] = 1

y_rotated_cropped[770:1155] = 2



pickle_out = open(os.path.join(DATAPATH, 'X_rotated_cropped_60.pickle'), 'wb')
pickle.dump(X_rotated_cropped, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATAPATH, 'y_rotated_cropped_60.pickle'), 'wb')
pickle.dump(y_rotated_cropped, pickle_out)
pickle_out.close()
