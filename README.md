# README

## Workflow

This is a handwritten digit recognition project with a GUI, divided into two modules:

* model training (I've built 3 different Stat-of-the-art model), **LeNet-5, CapsNet, BMCNNwHFCs**, and another simple CNN model **NDOCNN** with two dropout layers
* GUI module

this is the workflow and what I did.

![image-20250502102820851](C:\Users\29192\AppData\Roaming\Typora\typora-user-images\image-20250502102820851.png)

the key structure of the work:

```txt
DigitRecognition
├── ASSETS
├── C++
├── data
├── MODEL
│   ├── logs
│   ├── capslayer.py
│   └── model.py
├── python
│   ├── etc
│   ├── train.py
│   └── models
│       └── SmallImageBranchingMerging.py
├── result
├── draw.py
├── GUI.py
├── train.py
└── utils.py
```

## Model Training

First, you need to clone this project, then choose the model you want to train. It is recommended to use the model training method defined in `GUI.py`. You just need to uncomment the corresponding model.

When the model is instantiated, it will load the `.h5` file from the` MODEL/` directory (pre-trained parameters). If the file is not found, it will retrain the model.

### NDOCNN

```python
# model_path = "MODEL/cnn_model.h5"
# Initialize the NDOCNN model with input shape and model path
model = LeNet(x_train.shape[1:])
    model.summary()

# If model is not loaded, compile and train it
if not model.is_load:
    # model.train(x_train,y_train,epochs=20,batch_size=32)
    model.train(x_train, y_train, epochs=100, x_test=x_test, y_test=y_test)
    # model.save()  # Save the trained model

```

### LeNet-5

You need to first transform the input to a size of 32x32 If you are using LeNet-5.

```python
# LeNet-5
# Resize to 32×32 for LeNet-5
x_train = tf.image.resize(x_train, [32, 32]).numpy()
x_test = tf.image.resize(x_test, [32, 32]).numpy()
    
model = LeNet(x_train.shape[1:])
    model.summary()

# If model is not loaded, compile and train it
if not model.is_load:
    # model.train(x_train,y_train,epochs=20,batch_size=32)
    model.train(x_train, y_train, epochs=100, x_test=x_test, y_test=y_test)
    # model.save()  # Save the trained model
```

### CapsNet

You need to convert the training data to one-hot encoding to adapt to CapsNet's Margin loss.

```python
# CapsNet
# Convert the labels to one-hot encoding　only　caps　need
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 10 classes
y_test = tf.keras.utils.to_categorical(y_test, 10)    # 10 classes

model = CapsNet(x_train.shape[1:])
    model.summary()

# If model is not loaded, compile and train it
if not model.is_load:
    # model.train(x_train,y_train,epochs=20,batch_size=32)
    model.train(x_train, y_train, epochs=100, x_test=x_test, y_test=y_test)
    # model.save()  # Save the trained model
```

### BMCNNwHFCs

For BMCNNwHGCs, you can directly use the pre-trained model parameters, or refer to the original author's documentation. https://github.com/AdamByerly/BMCNNwHFCs

I've trained the model for 100 epoch and the `ckpt_path` stored the model parameter information.

```python
ckpt_path = './MODEL/logs/20250428134519/best_top1-77'
model = BMCNNwHFCs(ckpt_path)
```

## Evaluation

These figures show the loss convergence and accuracy values of four models.

![epoch-loss](D:\qi\paper\assignments\大二课设\result\epoch-loss.png)

![acc](D:\qi\paper\assignments\大二课设\result\acc.png)

## Bibliography

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278-2324.
* Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. *Advances in neural information processing systems*, *30*.
* Byerly, A., Kalganova, T., & Dear, I. (2021). No routing needed between capsules. *Neurocomputing*, *463*, 545-553.
* https://github.com/surya-veer/RealTime-DigitRecognition/tree/master
* https://github.com/AdamByerly/BMCNNwHFCs
* https://youtu.be/u3FLVbNn9Os?si=PEtiT_YIIOkVcT4t

Thank you for your contribution!
