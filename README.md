Assignment:2

1. Logs for 20 epochs
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 17s 289us/step - loss: 0.5417 - acc: 0.8477 - val_loss: 0.1168 - val_acc: 0.9755
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 9s 150us/step - loss: 0.2573 - acc: 0.9201 - val_loss: 0.0656 - val_acc: 0.9861
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 9s 146us/step - loss: 0.2004 - acc: 0.9386 - val_loss: 0.0526 - val_acc: 0.9885
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 9s 152us/step - loss: 0.1711 - acc: 0.9452 - val_loss: 0.0367 - val_acc: 0.9912
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 9s 144us/step - loss: 0.1524 - acc: 0.9482 - val_loss: 0.0308 - val_acc: 0.9925
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 9s 156us/step - loss: 0.1439 - acc: 0.9505 - val_loss: 0.0332 - val_acc: 0.9910
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 9s 148us/step - loss: 0.1317 - acc: 0.9530 - val_loss: 0.0288 - val_acc: 0.9918
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 9s 151us/step - loss: 0.1246 - acc: 0.9530 - val_loss: 0.0284 - val_acc: 0.9917
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 8s 139us/step - loss: 0.1176 - acc: 0.9544 - val_loss: 0.0285 - val_acc: 0.9918
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 9s 150us/step - loss: 0.1141 - acc: 0.9547 - val_loss: 0.0237 - val_acc: 0.9925
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 9s 149us/step - loss: 0.1109 - acc: 0.9558 - val_loss: 0.0216 - val_acc: 0.9938
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 9s 155us/step - loss: 0.1062 - acc: 0.9558 - val_loss: 0.0220 - val_acc: 0.9927
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 8s 141us/step - loss: 0.1055 - acc: 0.9559 - val_loss: 0.0223 - val_acc: 0.9930
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 9s 149us/step - loss: 0.1037 - acc: 0.9554 - val_loss: 0.0221 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 9s 147us/step - loss: 0.0994 - acc: 0.9575 - val_loss: 0.0220 - val_acc: 0.9930
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0994 - acc: 0.9565 - val_loss: 0.0200 - val_acc: 0.9935
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0979 - acc: 0.9567 - val_loss: 0.0220 - val_acc: 0.9936
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 9s 152us/step - loss: 0.0951 - acc: 0.9579 - val_loss: 0.0193 - val_acc: 0.9940
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 9s 148us/step - loss: 0.0955 - acc: 0.9573 - val_loss: 0.0191 - val_acc: 0.9942
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0930 - acc: 0.9580 - val_loss: 0.0193 - val_acc: 0.9939
<keras.callbacks.History at 0x7f6fc2f0d160>

2. result of model.evaluate (on test data)
[0.019294623884861358, 0.9939]

3. Strategy taken:
I should have a maxpooling layer where I can compress but I shouldn't have a maxpool layer close to my final output or last layer. Will increase my no. of channels using 1X1 then reduce them that's kind of expected model to build. Will use batch Normalization (to increase scaling and reduce shift i.e. make kernel give correct amplitude with which model can work. Hence every single channel is going to take equal part ensuring every channel which has a feature is loud enough to be visible) except for my last convolution layer asthat's last layer and we don't want to filter any information in last layer which should see complete object in order to make model predict it correctly. In order to pass out every single information that last layer has. It is layer which has to make final decison or call. Plus batch Normalization will create smaller gaps in bigger.Will also add dropout with small value except for last layer i.e. kind of regularisation to solve any overfitting related issue b/w accuracy and validation accuracy. Hence gap b/w test and training dataset will be reduced.
