# LOG of MODELS

## models

### (NDOCNN)normal  drop out CNN model

cite https://github.com/surya-veer/RealTime-DigitRecognition/tree/master

code

```python
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        self.model = model
        self.compile()
```

framework

```txt
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 26, 26, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 24, 24, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_1 (Activation)       │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 12, 12, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 12, 12, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 9216)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │     1,179,776 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_2 (Activation)       │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_3 (Activation)       │ (None, 10)             │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,199,882 (4.58 MB)
 Trainable params: 1,199,882 (4.58 MB)
 Non-trainable params: 0 (0.00 B)
```

## LeNet-5

```python
def build_model(self, input_shape):
    model = Sequential()

    # C1 - Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

    # S2 - Subsampling Layer (Average Pooling)
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # C3 - Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))

    # S4 - Subsampling Layer (Average Pooling)
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # C5 - Fully Connected Layer
    model.add(Dense(120, activation='relu'))

    # F6 - Fully Connected Layer
    model.add(Dense(84, activation='relu'))

    # Output Layer
    model.add(Dense(10, activation='softmax'))

    self.model = model
    self.compile()
```

```python
x_train, x_test = imageNormalization(x_train, x_test,mean=0.1037, std=0.3081)
# Resize 到 32×32 for LeNet-5
 # x_train = tf.image.resize(x_train, [32, 32]).numpy()
    # x_test = tf.image.resize(x_test, [32, 32]).numpy()
model.train(x_train,y_train,epochs=20,batch_size=32)
```

framework

```
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 28, 28, 6)      │           156 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d               │ (None, 14, 14, 6)      │             0 │
│ (AveragePooling2D)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 10, 10, 16)     │         2,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_1             │ (None, 5, 5, 16)       │             0 │
│ (AveragePooling2D)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 400)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 120)            │        48,120 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 84)             │        10,164 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │           850 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 61,706 (241.04 KB)
 Trainable params: 61,706 (241.04 KB)
 Non-trainable params: 0 (0.00 B)
```

result

```txt
Test loss on test samples:  0.03654353693127632
Validation accuracy:  0.9908000230789185
```

BMCNNwHFCs

```txt
top1: 0.9974899291992188
                      - loss: 0.01015567034482956
```

ff[^1]

[^1]: dfadf

$P(C|X)=\arg\max_{i}\left\| Cap_i \right\|$
