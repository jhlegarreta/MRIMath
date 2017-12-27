
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard
import os
import cv2
from EmailHandler import EmailHandler

def get_im(path):
    path=path.decode()
    img = cv2.imread(path,0)
    return img

def load_data(training_directory, start, finish):
    X_train = []
    segments = [[] for _ in range(8)]
    print('Reading images')
    for j in range(start,finish):
        print('Reading Patient ' + str(j))
        if j < 10:
            directory = os.fsencode(training_directory + '/Patient_(00' + str(j)  + ')_data/')
        elif j < 100:
            directory = os.fsencode(training_directory + '/Patient_(0' + str(j)  + ')_data/')

        else:
            directory = os.fsencode(training_directory + '/Patient_(' + str(j)  + ')_data/')
        for file in os.listdir(directory + b'/Original_Img_Data'):
            img = get_im(directory+b'/Original_Img_Data/'+file)
            X_train.append(img)
        segment_directory = os.fsencode(directory + b'Segmented_Img_Data')
        for dir in os.listdir(segment_directory):
            for file in os.listdir(segment_directory+b'/'+dir):
                ind = file[4:5]
                segments[int(ind.decode())-1].append(get_im(segment_directory+b'/'+dir+b'/'+file));

    return X_train, segments
    
emailHandler = EmailHandler()
emailHandler.prepareMessage("Testing", "Testing")
emailHandler.sendMessage("danielenricocahall@gmail.com")


input_img = Input(shape=(240, 240,1))  

F = 3
S = 2;

x = Conv2D(105, (F, F), activation='relu', padding='same')(input_img)
x = Conv2D(75, (F, F), activation='relu', padding='same')(x)
x = Conv2D(45, (F, F), activation='relu', padding='same')(x)
x = Conv2D(15, (F, F), activation='relu', padding='same')(x)
encoded = MaxPooling2D((S, S), padding='same')(x)
x = UpSampling2D((S, S))(encoded)
x = Conv2D(15, (F, F), activation='relu', padding='same')(x)
x = Conv2D(45, (F, F), activation='relu', padding='same')(x)
x = Conv2D(75, (F, F), activation='relu', padding='same')(x)
x = Conv2D(105, (F, F), activation='relu', padding='same')(x)
decoded = Conv2D(1, (F, F), activation='relu', padding='same')(x)

training, segments = load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images', 1, 85)

testing, segments2 = load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images',85,107)

segmentation_bank = [[] for _ in range(8)]
for i in range(1,8):
    print('Training network: ' + str(i))
    segmentation_bank[i] = Model(input_img, decoded)
    segmentation_bank[i].compile(optimizer='nadam', loss='mean_squared_error')
    n_imgs = len(training)
    training = np.array(training)
    training =training.reshape(n_imgs,240,240,1)
    training = training.astype('float32') / 255;
    segments[i] = np.array(segments[i]);
    segments[i] = segments[i].reshape(n_imgs,240,240,1)
    n_imgs2 = len(testing)
    testing = np.array(testing)
    testing =testing.reshape(n_imgs2,240,240,1)
    testing= testing.astype('float32') / 255;
    segments2[i] = np.array(segments2[i]);
    segments2[i] = segments2[i].reshape(n_imgs2,240,240,1)
    segmentation_bank[i].fit(training, segments[i],
                epochs=30,
                batch_size=50,
                shuffle=True,
                validation_data=(testing, segments2[i]),
                callbacks=[TensorBoard(log_dir='/tmp/segment_data')])
    # serialize model to JSON
    segmentation_bank[i].save('/coe_data/MRIMath/MS_Research/model_' + str(i) +'.h5')








