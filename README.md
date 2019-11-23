Assignment:2

1. Logs for 20 epochs
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 14s 232us/step - loss: 0.2092 - acc: 0.9348 - val_loss: 0.0713 - val_acc: 0.9767
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0624 - acc: 0.9797 - val_loss: 0.0392 - val_acc: 0.9873
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 7s 115us/step - loss: 0.0474 - acc: 0.9854 - val_loss: 0.0341 - val_acc: 0.9895
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0409 - acc: 0.9869 - val_loss: 0.0294 - val_acc: 0.9908
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0357 - acc: 0.9886 - val_loss: 0.0283 - val_acc: 0.9912
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0336 - acc: 0.9893 - val_loss: 0.0254 - val_acc: 0.9920
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0305 - acc: 0.9902 - val_loss: 0.0263 - val_acc: 0.9916
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 7s 119us/step - loss: 0.0281 - acc: 0.9908 - val_loss: 0.0201 - val_acc: 0.9935
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0258 - acc: 0.9920 - val_loss: 0.0226 - val_acc: 0.9939
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0249 - acc: 0.9923 - val_loss: 0.0229 - val_acc: 0.9938
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0241 - acc: 0.9922 - val_loss: 0.0217 - val_acc: 0.9933
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0231 - acc: 0.9926 - val_loss: 0.0231 - val_acc: 0.9930
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0227 - acc: 0.9926 - val_loss: 0.0224 - val_acc: 0.9941
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0208 - acc: 0.9927 - val_loss: 0.0199 - val_acc: 0.9934
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0195 - acc: 0.9935 - val_loss: 0.0186 - val_acc: 0.9933
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0184 - acc: 0.9938 - val_loss: 0.0210 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 7s 117us/step - loss: 0.0189 - acc: 0.9939 - val_loss: 0.0191 - val_acc: 0.9942
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0177 - acc: 0.9938 - val_loss: 0.0181 - val_acc: 0.9946
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0171 - acc: 0.9942 - val_loss: 0.0199 - val_acc: 0.9938
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 7s 116us/step - loss: 0.0168 - acc: 0.9943 - val_loss: 0.0194 - val_acc: 0.9948
<keras.callbacks.History at 0x7f8c270dc8d0>

2. result of model.evaluate (on test data)
[0.019371607353230503, 0.9948]

3. Strategy taken:
I should have a maxpooling layer where I can compress but I shouldn't have a maxpool layer close to my final output or last layer. Will increase my no. of channels using 1X1 then reduce them that's kind of expected model to build. Will use batch Normalization (to increase scaling and reduce shift i.e. make kernel give correct amplitude with which model can work. Hence every single channel is going to take equal part ensuring every channel which has a feature is loud enough to be visible) except for my last convolution layer asthat's last layer and we don't want to filter any information in last layer which should see complete object in order to make model predict it correctly. In order to pass out every single information that last layer has. It is layer which has to make final decison or call. Plus batch Normalization will create smaller gaps in bigger.Will also add dropout with small value except for last layer i.e. kind of regularisation to solve any overfitting related issue b/w accuracy and validation accuracy. Hence gap b/w test and training dataset accuracy will be reduced. Also using a Learning rate scheduler can try to control the feedback giving to every neuron after a batch.
