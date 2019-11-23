Assignment:2

1. Logs for 20 epochs
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 18s 298us/step - loss: 0.0742 - acc: 0.9774 - val_loss: 0.0596 - val_acc: 0.9806
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 9s 157us/step - loss: 0.0543 - acc: 0.9829 - val_loss: 0.0341 - val_acc: 0.9888
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 9s 156us/step - loss: 0.0449 - acc: 0.9857 - val_loss: 0.0341 - val_acc: 0.9894
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 9s 156us/step - loss: 0.0394 - acc: 0.9878 - val_loss: 0.0251 - val_acc: 0.9921
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 9s 158us/step - loss: 0.0349 - acc: 0.9891 - val_loss: 0.0273 - val_acc: 0.9907
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 10s 159us/step - loss: 0.0317 - acc: 0.9896 - val_loss: 0.0274 - val_acc: 0.9915
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 9s 152us/step - loss: 0.0298 - acc: 0.9903 - val_loss: 0.0292 - val_acc: 0.9910
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 10s 159us/step - loss: 0.0283 - acc: 0.9912 - val_loss: 0.0231 - val_acc: 0.9921
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 9s 156us/step - loss: 0.0284 - acc: 0.9911 - val_loss: 0.0193 - val_acc: 0.9936
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 10s 160us/step - loss: 0.0262 - acc: 0.9920 - val_loss: 0.0275 - val_acc: 0.9926
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 9s 156us/step - loss: 0.0234 - acc: 0.9926 - val_loss: 0.0262 - val_acc: 0.9923
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 10s 159us/step - loss: 0.0231 - acc: 0.9927 - val_loss: 0.0231 - val_acc: 0.9922
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 9s 148us/step - loss: 0.0225 - acc: 0.9928 - val_loss: 0.0208 - val_acc: 0.9937
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0220 - acc: 0.9926 - val_loss: 0.0209 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 9s 158us/step - loss: 0.0203 - acc: 0.9935 - val_loss: 0.0214 - val_acc: 0.9935
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 10s 161us/step - loss: 0.0207 - acc: 0.9933 - val_loss: 0.0209 - val_acc: 0.9941
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 9s 154us/step - loss: 0.0192 - acc: 0.9939 - val_loss: 0.0214 - val_acc: 0.9934
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 10s 160us/step - loss: 0.0188 - acc: 0.9940 - val_loss: 0.0214 - val_acc: 0.9934
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 9s 149us/step - loss: 0.0184 - acc: 0.9940 - val_loss: 0.0191 - val_acc: 0.9939
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 10s 159us/step - loss: 0.0191 - acc: 0.9936 - val_loss: 0.0202 - val_acc: 0.9938
<keras.callbacks.History at 0x7f6f941436a0>

2. result of model.evaluate (on test data)
[0.020194914594515286, 0.9938]

3. Strategy taken:
I should have a maxpooling layer where I can compress but I shouldn't have a maxpool layer close to my final output or last layer. Will increase my no. of channels using 1X1 then reduce them that's kind of expected model to build. Will use batch Normalization (to increase scaling and reduce shift i.e. make kernel give correct amplitude with which model can work. Hence every single channel is going to take equal part ensuring every channel which has a feature is loud enough to be visible) except for my last convolution layer asthat's last layer and we don't want to filter any information in last layer which should see complete object in order to make model predict it correctly. In order to pass out every single information that last layer has. It is layer which has to make final decison or call. Plus batch Normalization will create smaller gaps in bigger.Will also add dropout with small value except for last layer i.e. kind of regularisation to solve any overfitting related issue b/w accuracy and validation accuracy. Hence gap b/w test and training dataset accuracy will be reduced. Also using a Learning rate scheduler can try to control the feedback giving to every neuron after a batch.
