# import keras
from android.os import Environment
from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras .callbacks import Callback
import time
import os
import gc
import random
import cv2
import numpy as np
# numt=0
class TimeStop(Callback):
    def __init__(self,seconds=0):
        super(Callback,self).__init__()
        self.start_time=0
        self.seconds=seconds

    def on_train_begin(self, logs=None):
        self.start_time=time.time()
    def on_batch_end(self, batch, logs=None):
        print(time.time()-self.start_time)
        if time.time()-self.start_time>self.seconds:
            self.model.stop_training=True
            print("Stopped after %s seconds"%(self.seconds))

def main(second=150,conv=2,dens=2 ):
    print(tf.__version__)
    keras.backend.clear_session()
    tf.reset_default_graph()
    graph=tf.get_default_graph()
    with graph.as_default():
        print(second)
        batch_size=32
        classes=10
        epochs=5
        img_rows, img_cols = 28, 28

        (xtr,ytr),(xtst,ytst)=mnist.load_data()

        xtr = xtr.reshape(xtr.shape[0], img_rows, img_cols, 1)
        xtst = xtst.reshape(xtst.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        ytr=keras.utils.to_categorical(ytr,classes)
        ytst=keras.utils.to_categorical(ytst,classes)

        model= Sequential()
        model.add(Conv2D(32,(3,3),input_shape=input_shape))
        model.add(Activation('relu'))
        for aa in range(0,conv-1):
            model.add(Conv2D(32,(3,3)))
            model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # model.add(Conv2D(64,(3,3)))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64,(3,3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        for bb in range(1, dens):
            model.add(Dense(512//bb))
            model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        # global numt
        # opt=keras.optimizers.Adam(lr=.0001,decay=1e-7)
        # optimizers=["adam","nadam","adamax"]
        model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
        # numt+=1

        xtr=xtr.astype('float32')
        xtst=xtst.astype('float32')
        xtr/=255
        xtst/=255

        stopper=TimeStop(seconds=second)

        model.fit(xtr,ytr,batch_size=batch_size,epochs=epochs,validation_data=(xtst,ytst),shuffle=True,callbacks=[stopper])
        randomSam=random.randint(0,len(xtst)-1000)
        scores=model.evaluate(xtst[randomSam:randomSam+1000],ytst[randomSam:randomSam+1000],verbose=1)
        print("Loss:",scores[0] )
        print("Accuracy:",scores[1])
        model.save('model.h5')
        del model
        gc.collect()
        # keras.backend.clear_session()
        # tf.reset_default_graph()
        # graph = tf.get_default_graph()

        return scores[1]



def run(filename):
    img_rows, img_cols = 28, 28
    new_model=tf.keras.models.load_model("model.h5")
    img=cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    cv2.imshow("ayo",img)
    cv2.waitKey(0)
    print('here')
    print(img.shape)
    img=np.asarray(img)
    print(img.shape)

    img=np.expand_dims(img,axis=0)
    img=np.expand_dims(img,axis=4)
    # classes = 10
    # (xtr, ytr), (xtst, ytst) = mnist.load_data()
    #
    # xtr = xtr.reshape(xtr.shape[0], img_rows, img_cols, 1)
    # xtst = xtst.reshape(xtst.shape[0], img_rows, img_cols, 1)
    # randomSam = random.randint(0, len(xtst) - 1000)
    # ytr = keras.utils.to_categorical(ytr, classes)
    # ytst = keras.utils.to_categorical(ytst, classes)
    # scores = new_model.evaluate(xtst[randomSam:randomSam + 1000], ytst[randomSam:randomSam + 1000], verbose=1)
    # print("Loss:", scores[0])
    # print("Accuracy:", scores[1])
    # print( new_model.summary())
    result=new_model.predict(img,1)
    print(result)
    print(np.where(result==1)[1][0])


#main(10, 2,2)
#run("C:\\Users\\adity\\Documents\\mn.png")
# main(10,2,3)
#C:\Users\adity\Downloads\test.jpg

