1. Final Validation accuracy for Base Network

Epoch 50/50
390/390 [==============================] - 7s 17ms/step - loss: 0.3212 - acc: 0.8919 - val_loss: 0.5962 - val_acc: 0.8246
Model took 348.86 seconds to train

Accuracy on test data is: 82.46

2. Your model definition (model.add... ) with output channel size and receptive field

# Create the model

import keras
from keras.layers import SeparableConv2D, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D
model = Sequential()
# model.add(SeparableConv1D(48, 3, padding='valid', input_shape=(32, 3), 
#                           data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation='relu', 
#                           use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', 
#                           bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, 
#                           bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, 
#                           pointwise_constraint=None, bias_constraint=None))

train_features = np.random.rand(32, 32, 3)

model.add(SeparableConv1D(64, 3, padding='same', input_shape=(train_features.shape[1],train_features.shape[2])))
model.add(Activation('relu'))

model.add(SeparableConv1D(32, 3, padding='same', activation='relu' ))

model.add(GlobalAveragePooling1D())
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

Model: "sequential_108"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv1d_207 (Separa (None, 30, 48)            201       
_________________________________________________________________
activation_90 (Activation)   (None, 30, 48)            0         
_________________________________________________________________
separable_conv1d_208 (Separa (None, 28, 48)            2496      
_________________________________________________________________
activation_91 (Activation)   (None, 28, 48)            0         
_________________________________________________________________
max_pooling1d_45 (MaxPooling (None, 14, 48)            0         
_________________________________________________________________
separable_conv1d_209 (Separa (None, 12, 96)            4848      
_________________________________________________________________
separable_conv1d_210 (Separa (None, 10, 96)            9600      
_________________________________________________________________
global_average_pooling1d_5 ( (None, 96)                0         
_________________________________________________________________
batch_normalization_109 (Bat (None, 96)                384       
_________________________________________________________________
dropout_206 (Dropout)        (None, 96)                0         
_________________________________________________________________
dense_113 (Dense)            (None, 64)                6208      
_________________________________________________________________
activation_92 (Activation)   (None, 64)                0         
_________________________________________________________________
dropout_207 (Dropout)        (None, 64)                0         
_________________________________________________________________
dense_114 (Dense)            (None, 10)                650       
=================================================================
Total params: 24,387
Trainable params: 24,195
Non-trainable params: 192
_____________________________

3. Your 50 epoch logs

Epoch 1/50
390/390 [==============================] - 23s 59ms/step - loss: 1.8897 - acc: 0.2706 - val_loss: 1.4979 - val_acc: 0.4453
Epoch 2/50
390/390 [==============================] - 20s 52ms/step - loss: 1.3654 - acc: 0.5035 - val_loss: 1.1105 - val_acc: 0.5974
Epoch 3/50
390/390 [==============================] - 20s 52ms/step - loss: 1.0934 - acc: 0.6147 - val_loss: 1.0536 - val_acc: 0.6424
Epoch 4/50
390/390 [==============================] - 20s 52ms/step - loss: 0.9311 - acc: 0.6795 - val_loss: 0.8866 - val_acc: 0.6950
Epoch 5/50
390/390 [==============================] - 20s 52ms/step - loss: 0.8344 - acc: 0.7131 - val_loss: 0.7482 - val_acc: 0.7413
Epoch 6/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7645 - acc: 0.7396 - val_loss: 0.7445 - val_acc: 0.7434
Epoch 7/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7128 - acc: 0.7587 - val_loss: 0.7101 - val_acc: 0.7595
Epoch 8/50
390/390 [==============================] - 21s 53ms/step - loss: 0.6753 - acc: 0.7701 - val_loss: 0.6664 - val_acc: 0.7759
Epoch 9/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6298 - acc: 0.7876 - val_loss: 0.6599 - val_acc: 0.7811
Epoch 10/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6025 - acc: 0.7931 - val_loss: 0.6392 - val_acc: 0.7863
Epoch 11/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5854 - acc: 0.8012 - val_loss: 0.6337 - val_acc: 0.7864
Epoch 12/50
390/390 [==============================] - 21s 53ms/step - loss: 0.5539 - acc: 0.8128 - val_loss: 0.5840 - val_acc: 0.8021
Epoch 13/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5383 - acc: 0.8175 - val_loss: 0.5797 - val_acc: 0.8089
Epoch 14/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5148 - acc: 0.8249 - val_loss: 0.6210 - val_acc: 0.7963
Epoch 15/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5072 - acc: 0.8299 - val_loss: 0.6041 - val_acc: 0.7979
Epoch 16/50
390/390 [==============================] - 21s 53ms/step - loss: 0.4757 - acc: 0.8386 - val_loss: 0.6152 - val_acc: 0.7949
Epoch 17/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4730 - acc: 0.8395 - val_loss: 0.5865 - val_acc: 0.8106
Epoch 18/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4631 - acc: 0.8441 - val_loss: 0.5877 - val_acc: 0.8086
Epoch 19/50
390/390 [==============================] - 21s 53ms/step - loss: 0.4460 - acc: 0.8491 - val_loss: 0.5904 - val_acc: 0.8082
Epoch 20/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4335 - acc: 0.8523 - val_loss: 0.5995 - val_acc: 0.8078
Epoch 21/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4267 - acc: 0.8556 - val_loss: 0.5812 - val_acc: 0.8127
Epoch 22/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4179 - acc: 0.8579 - val_loss: 0.6436 - val_acc: 0.7973
Epoch 23/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4107 - acc: 0.8618 - val_loss: 0.6047 - val_acc: 0.8070
Epoch 24/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4053 - acc: 0.8641 - val_loss: 0.6063 - val_acc: 0.8097
Epoch 25/50
390/390 [==============================] - 20s 53ms/step - loss: 0.3972 - acc: 0.8651 - val_loss: 0.5740 - val_acc: 0.8150
Epoch 26/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3855 - acc: 0.8700 - val_loss: 0.5631 - val_acc: 0.8180
Epoch 27/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3797 - acc: 0.8722 - val_loss: 0.5680 - val_acc: 0.8165
Epoch 28/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3711 - acc: 0.8737 - val_loss: 0.5726 - val_acc: 0.8192
Epoch 29/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3617 - acc: 0.8748 - val_loss: 0.5610 - val_acc: 0.8241
Epoch 30/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3586 - acc: 0.8791 - val_loss: 0.5922 - val_acc: 0.8167
Epoch 31/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3596 - acc: 0.8775 - val_loss: 0.6058 - val_acc: 0.8153
Epoch 32/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3549 - acc: 0.8797 - val_loss: 0.5777 - val_acc: 0.8227
Epoch 33/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3505 - acc: 0.8825 - val_loss: 0.5837 - val_acc: 0.8223
Epoch 34/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3400 - acc: 0.8851 - val_loss: 0.5824 - val_acc: 0.8213
Epoch 35/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3308 - acc: 0.8880 - val_loss: 0.6120 - val_acc: 0.8174
Epoch 36/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3307 - acc: 0.8895 - val_loss: 0.5903 - val_acc: 0.8267
Epoch 37/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3319 - acc: 0.8887 - val_loss: 0.5996 - val_acc: 0.8254
Epoch 38/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3247 - acc: 0.8927 - val_loss: 0.5802 - val_acc: 0.8243
Epoch 39/50
390/390 [==============================] - 21s 53ms/step - loss: 0.3240 - acc: 0.8920 - val_loss: 0.5740 - val_acc: 0.8251
Epoch 40/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3107 - acc: 0.8963 - val_loss: 0.6383 - val_acc: 0.8219
Epoch 41/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3118 - acc: 0.8946 - val_loss: 0.5840 - val_acc: 0.8294
Epoch 42/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3088 - acc: 0.8983 - val_loss: 0.6349 - val_acc: 0.8189
Epoch 43/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3114 - acc: 0.8979 - val_loss: 0.6165 - val_acc: 0.8262
Epoch 44/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3079 - acc: 0.8973 - val_loss: 0.6103 - val_acc: 0.8241
Epoch 45/50
390/390 [==============================] - 20s 52ms/step - loss: 0.2972 - acc: 0.9014 - val_loss: 0.6025 - val_acc: 0.8281
Epoch 46/50
390/390 [==============================] - 20s 52ms/step - loss: 0.2980 - acc: 0.8999 - val_loss: 0.5861 - val_acc: 0.8265
Epoch 47/50
390/390 [==============================] - 21s 53ms/step - loss: 0.3045 - acc: 0.8989 - val_loss: 0.5984 - val_acc: 0.8301
Epoch 48/50
390/390 [==============================] - 20s 52ms/step - loss: 0.2928 - acc: 0.9031 - val_loss: 0.6006 - val_acc: 0.8269
Epoch 49/50
390/390 [==============================] - 20s 52ms/step - loss: 0.2917 - acc: 0.9033 - val_loss: 0.5943 - val_acc: 0.8304
Epoch 50/50
390/390 [==============================] - 20s 52ms/step - loss: 0.2901 - acc: 0.9050 - val_loss: 0.5728 - val_acc: 0.8344
Model took 1021.03 seconds to train

Accuracy on test data is: 83.44