"""
Output from run (2023/01/21)

loading data
loaded data: (95154, 2, 24, 32)
Testing on trial 1, training on all others
  Train:  81k, Test:  14k
  Class balance: train 52% | 48%, test  52% | 48%
Train accuracy: 99.60%
Test accuracy:  96.49%

Testing on trial 2, training on all others
  Train:  78k, Test:  17k
  Class balance: train 57% | 43%, test  28% | 72%
Train accuracy: 99.65%
Test accuracy:  96.30%

Testing on trial 3, training on all others
  Train:  53k, Test:  42k
  Class balance: train 52% | 48%, test  52% | 48%
Train accuracy: 99.72%
Test accuracy:  97.76%

Testing on trial 4, training on all others
  Train:  73k, Test:  22k
  Class balance: train 47% | 53%, test  71% | 29%
Train accuracy: 99.69%
Test accuracy:  97.81%

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 22, 32)        608       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 15, 11, 32)       0         
 )                                                           
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 9, 64)         18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 4, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1536)              0         
                                                                 
 dense (Dense)               (None, 512)               786944    
                                                                 
 dense_1 (Dense)             (None, 2)                 1026      
                                                                 
=================================================================
Total params: 807,074
Trainable params: 807,074
Non-trainable params: 0
_________________________________________________________________
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
import tonic
from sklearn.preprocessing import StandardScaler


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.chdir('dvs_flow_github')


data_folder = '../data/bin_1ms_comp'
chars = list("AB")
fidxs = list(range(1, 5))

print('loading data')
data = []
label = []
trial = []
for class_idx, ch in enumerate(chars):
    for fi in fidxs:
        path = f'{data_folder}/{ch}{fi}.npy'
        d = np.load(path, allow_pickle=True)
        d = d[[e.shape[0] >= 1_000 for e in d]]
        data.append(d)
        label.append([class_idx] * len(d))
        trial.append([fi] * len(d))
data = np.concatenate(data)
label = np.concatenate(label)
trial = np.concatenate(trial)
transform = tonic.transforms.ToImage(sensor_size=(32, 24, 2,))
data = np.array([transform(img) for img in data])
print('loaded data:', data.shape)

# training

tracc = []
teacc = []
for test_trial in range(1, 5):
    # train test split
    train_idxs = trial != test_trial
    test_idxs = trial == test_trial
    train_data = data[train_idxs]
    train_labels = label[train_idxs]
    train_trials = trial[train_idxs]
    test_data = data[test_idxs]
    test_labels = label[test_idxs]
    test_trials = trial[test_idxs]

    # log
    print(f'Testing on trial {test_trial}, training on all others')
    print(f'  Train: {train_data.shape[0]/1000:3.0f}k, Test: {test_data.shape[0]/1000:3.0f}k')
    ntot = np.sum(train_labels == 0) + np.sum(train_labels == 1)
    print('  Class balance: ', end='')
    print(f'train {np.sum(train_labels == 0)/ntot:.0%} | {np.sum(train_labels == 1)/ntot:.0%}', end=', ')
    ntot = np.sum(test_labels == 0) + np.sum(test_labels == 1)
    print(f'test  {np.sum(test_labels == 0)/ntot:.0%} | {np.sum(test_labels == 1)/ntot:.0%}')

    # pre-processing
    scaler = StandardScaler()
    tr_shape = train_data.shape
    te_shape = test_data.shape
    train_data_ = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1)).reshape(tr_shape).transpose(0, 3, 2, 1)
    test_data_ = scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(te_shape).transpose(0, 3, 2, 1)

    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(train_data_.shape[1:])),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(train_data_.shape[1:])),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_, keras.utils.to_categorical(train_labels), epochs=20, batch_size=64, verbose=1)
    train_score = model.evaluate(train_data_, keras.utils.to_categorical(train_labels))
    test_score = model.evaluate(test_data_, keras.utils.to_categorical(test_labels))
    tracc.append(train_score[1])
    teacc.append(test_score[1])
    print(f'Train accuracy: {tracc[-1]:.2%}')
    print(f'Test accuracy:  {teacc[-1]:.2%}')
    print()

print(f'training: {tracc.mean()/100:.2%} +- {tracc.std()/100:.2%}')
print(f'testing:  {teacc.mean()/100:.2%} +- {teacc.std()/100:.2%}')


"""
Full log:

Epoch 1/20
1265/1265 [==============================] - 11s 8ms/step - loss: 0.1188 - accuracy: 0.9591
Epoch 2/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0554 - accuracy: 0.9813
Epoch 3/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0429 - accuracy: 0.9853
Epoch 4/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0376 - accuracy: 0.9865
Epoch 5/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0308 - accuracy: 0.9892
Epoch 6/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0271 - accuracy: 0.9907
Epoch 7/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0223 - accuracy: 0.9919
Epoch 8/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0191 - accuracy: 0.9933
Epoch 9/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0192 - accuracy: 0.9933
Epoch 10/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0158 - accuracy: 0.9942
Epoch 11/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0144 - accuracy: 0.9948
Epoch 12/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0175 - accuracy: 0.9945
Epoch 13/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0118 - accuracy: 0.9959
Epoch 14/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0109 - accuracy: 0.9961
Epoch 15/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0109 - accuracy: 0.9965
Epoch 16/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0113 - accuracy: 0.9963
Epoch 17/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0103 - accuracy: 0.9968
Epoch 18/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0118 - accuracy: 0.9966
Epoch 19/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0058 - accuracy: 0.9979
Epoch 20/20
1265/1265 [==============================] - 10s 8ms/step - loss: 0.0088 - accuracy: 0.9970
2529/2529 [==============================] - 9s 4ms/step - loss: 0.0114 - accuracy: 0.9960  
445/445 [==============================] - 2s 4ms/step - loss: 0.3758 - accuracy: 0.9649

--------------------
--------------------
--------------------

Epoch 1/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.1127 - accuracy: 0.9617
Epoch 2/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0517 - accuracy: 0.9822
Epoch 3/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0409 - accuracy: 0.9858
Epoch 4/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0332 - accuracy: 0.9884
Epoch 5/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0295 - accuracy: 0.9902
Epoch 6/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0259 - accuracy: 0.9916
Epoch 7/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0205 - accuracy: 0.9933
Epoch 8/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0194 - accuracy: 0.9932
Epoch 9/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0161 - accuracy: 0.9943
Epoch 10/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0157 - accuracy: 0.9945
Epoch 11/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0135 - accuracy: 0.9957
Epoch 12/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0135 - accuracy: 0.9956
Epoch 13/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0136 - accuracy: 0.9953
Epoch 14/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0097 - accuracy: 0.9967
Epoch 15/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0093 - accuracy: 0.9968
Epoch 16/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0096 - accuracy: 0.9964
Epoch 17/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0101 - accuracy: 0.9965
Epoch 18/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0068 - accuracy: 0.9977
Epoch 19/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0077 - accuracy: 0.9972
Epoch 20/20
1223/1223 [==============================] - 10s 8ms/step - loss: 0.0118 - accuracy: 0.9967
2446/2446 [==============================] - 9s 4ms/step - loss: 0.0094 - accuracy: 0.9965
529/529 [==============================] - 2s 4ms/step - loss: 0.1751 - accuracy: 0.9630

--------------------
--------------------
--------------------

Epoch 1/20
829/829 [==============================] - 7s 8ms/step - loss: 0.1600 - accuracy: 0.9439
Epoch 2/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0761 - accuracy: 0.9737
Epoch 3/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0594 - accuracy: 0.9801
Epoch 4/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0517 - accuracy: 0.9821
Epoch 5/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0415 - accuracy: 0.9854
Epoch 6/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0370 - accuracy: 0.9868
Epoch 7/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0294 - accuracy: 0.9898
Epoch 8/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0287 - accuracy: 0.9905
Epoch 9/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0251 - accuracy: 0.9911
Epoch 10/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0214 - accuracy: 0.9932
Epoch 11/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0209 - accuracy: 0.9928
Epoch 12/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0176 - accuracy: 0.9940
Epoch 13/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0168 - accuracy: 0.9940
Epoch 14/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0110 - accuracy: 0.9961
Epoch 15/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0130 - accuracy: 0.9950
Epoch 16/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0150 - accuracy: 0.9948
Epoch 17/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0123 - accuracy: 0.9954
Epoch 18/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0104 - accuracy: 0.9964
Epoch 19/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0090 - accuracy: 0.9969
Epoch 20/20
829/829 [==============================] - 7s 8ms/step - loss: 0.0073 - accuracy: 0.9973
1657/1657 [==============================] - 6s 4ms/step - loss: 0.0075 - accuracy: 0.9972
1318/1318 [==============================] - 5s 4ms/step - loss: 0.3541 - accuracy: 0.9776 

--------------------
--------------------
--------------------

Epoch 1/20
1146/1146 [==============================] - 10s 8ms/step - loss: 0.1329 - accuracy: 0.9552
Epoch 2/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0606 - accuracy: 0.9799
Epoch 3/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0486 - accuracy: 0.9835
Epoch 4/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0418 - accuracy: 0.9862
Epoch 5/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0346 - accuracy: 0.9880
Epoch 6/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0300 - accuracy: 0.9899
Epoch 7/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0279 - accuracy: 0.9902
Epoch 8/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0248 - accuracy: 0.9914
Epoch 9/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0223 - accuracy: 0.9924
Epoch 10/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0216 - accuracy: 0.9928
Epoch 11/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0170 - accuracy: 0.9942
Epoch 12/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0153 - accuracy: 0.9945
Epoch 13/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0138 - accuracy: 0.9950
Epoch 14/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0136 - accuracy: 0.9952
Epoch 15/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0119 - accuracy: 0.9960
Epoch 16/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0113 - accuracy: 0.9962
Epoch 17/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0128 - accuracy: 0.9957
Epoch 18/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0104 - accuracy: 0.9965 
Epoch 19/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0083 - accuracy: 0.9970
Epoch 20/20
1146/1146 [==============================] - 9s 8ms/step - loss: 0.0125 - accuracy: 0.9963
2291/2291 [==============================] - 11s 5ms/step - loss: 0.0089 - accuracy: 0.9969
684/684 [==============================] - 3s 5ms/step - loss: 0.1123 - accuracy: 0.9781
"""
