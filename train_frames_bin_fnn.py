"""
Output from run (2023/01/20)

Testing on trial 1, training on all others
  Train:  81k, Test:  14k
  Class balance: train 52% | 48%, test  52% | 48%
Train accuracy: 99.78%
Test accuracy:  96.97%

Testing on trial 2, training on all others
  Train:  78k, Test:  17k
  Class balance: train 57% | 43%, test  28% | 72%
Train accuracy: 99.34%
Test accuracy:  97.70%

Testing on trial 3, training on all others
  Train:  53k, Test:  42k
  Class balance: train 52% | 48%, test  52% | 48%
Train accuracy: 99.25%
Test accuracy:  97.44%

Testing on trial 4, training on all others
  Train:  73k, Test:  22k
  Class balance: train 47% | 53%, test  71% | 29%
Train accuracy: 99.71%
Test accuracy:  97.92%

Training: 99.52 +- 0.23
Testing:  97.51 +- 0.35


trial 1
Train accuracy: 99.41%
Test accuracy:  96.42%

trial 2
Train accuracy: 99.73%
Test accuracy:  97.34%

trial 3
Train accuracy: 99.49%
Test accuracy:  97.64%

trial 4
Train accuracy: 99.19%
Test accuracy:  97.86%

np.array([99.41, 99.73, 99.49, 99.19]) 
  -> training: 99.46\% $\pm$ 0.20\%
np.array([96.42, 97.34, 97.64, 97.86]) 
  -> testing: 97.32\% \pm 0.55\%
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
import tonic
from sklearn.preprocessing import StandardScaler


data_folder = '../data/bin_1ms_comp/'
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
    train_data_ = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1))
    test_data_ = scaler.transform(test_data.reshape(test_data.shape[0], -1))

    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(train_data_.shape[1],)),
        # keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_, keras.utils.to_categorical(train_labels), epochs=10, batch_size=64, verbose=1)
    train_score = model.evaluate(train_data_, keras.utils.to_categorical(train_labels))
    test_score = model.evaluate(test_data_, keras.utils.to_categorical(test_labels))
    tracc.append(train_score[1])
    teacc.append(test_score[1])
    print(f'Train accuracy: {tracc[-1]:.2%}')
    print(f'Test accuracy:  {teacc[-1]:.2%}')
    print()

print(f'training: {tracc.mean()/100:.2%} +- {tracc.std()/100:.2%}')
print(f'testing:  {teacc.mean()/100:.2%} +- {teacc.std()/100:.2%}')
