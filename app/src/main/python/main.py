# import keras
from tensorflow.python import keras

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import os
def main():
    batch_size=32
    classes=10
    epochs=1
    img_rows, img_cols = 28, 28

    (xtr,ytr),(xtst,ytst)=mnist.load_data()

    xtr = xtr.reshape(xtr.shape[0], img_rows, img_cols, 1)
    xtst = xtst.reshape(xtst.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    ytr=keras.utils.to_categorical(ytr,classes)
    ytst=keras.utils.to_categorical(ytst,classes)

    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=input_shape))
    model.add(Activation('relu'))
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

    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    # opt=keras.optimizers.Adam(lr=.0001,decay=1e-7)
    
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    
    xtr=xtr.astype('float32')
    xtst=xtst.astype('float32')
    xtr/=255
    xtst/=255
    
    model.fit(xtr,ytr,batch_size=batch_size,epochs=epochs,validation_data=(xtst,ytst),shuffle=True)

    
    scores=model.evaluate(xtst,ytst,verbose=1)
    print("Loss:",scores[0] )
    print("Accuracy:",scores[1])
    return scores[1]


main()