# Using CNN in grayscale image edge/corner detection 
> Author: 侯太格  
Student ID: 1600012897
## Files and Directories
* `data_washing.py`  
    used to process the raw data into organized labled data, storing which in file `train_data.npy`
* `kt_utils.py`  
    used to load data
* `edge_training.py`  
    defined CNN to detect edge and store the model in file `edge_detec_model.h5`  
* `edge_test.py`  
    using trained CNN to detect edge in the test images
* `corner_training.py`   
    define CNN to detect corner and store the model in file `cor_detec_model.h5`  
* `corner_test.py`  
    using trained CNN to detect corner in the test images   
* `/edge_detection_test`  
    stored the edge image detected by CNN
* `/corner_detection_test`  
    stored the corner image detected by CNN
* `/synthetic_characters`  
    stored test images
## Guide
* to wash and label the data  
    `pyhton3 data_washing.py`
* to train edge detection model  
    `python3 edge_training.py`
* to get the edge images detected from test images      
    `python3 edge_test.py`
* to train corner detection model  
    `python3 corner_training.py`  
* to get the edge images detected from test images    
    `python3 corner_test.py`

## CNN Structure Introduction
### Edge Detect CNN
#### Convolution Layer 0  
* input  
    keras tensor shaped `(x, x, 1)`, x can be any interger 
* filters  
    8 filters, size = `(3, 3)`, stride = `(1, 1)`
* activation funtion: `relu  `
#### Convolution Layer 1
* input  
    keras tensor shaped `(x - 2, x - 2, 8)`, x can be any interger
* filters  
    1 filters, size = `(3, 3)`, stride = `(1, 1)`
* activation funtion: `sigmoid`       



### Corner Detect CNN
#### Convolution Layer 0  
* input  
    keras tensor shaped `(x, x, 1)`, x can be any interger 
* filters  
    8 filters, size = `(3, 3)`, stride = `(1, 1)`
* activation funtion: `relu`
#### Convolution Layer 1
* input  
    keras tensor shaped `(x - 2, x - 2, 8)`, x can be any interger
* filters  
    8 filters, size = `(1, 1)`, stride = `(1, 1)`
* activation funtion: `relu`     
#### Convolution Layer 2
* input  
    keras tensor shaped `(x - 2, x - 2, 8)`, x can be any interger
* filters  
    16 filters, size = `(3, 3)`, stride = `(1, 1)`
* activation funtion: `relu`     
#### Convolution Layer 3
* input  
    keras tensor shaped `(x - 4, x - 4, 16)`, x can be any interger
* filters  
    3 filters, size = `(1, 1)`, stride = `(1, 1)`
* activation funtion: `relu`    
#### Convolution Layer 4
* input  
    keras tensor shaped `(x - 4, x - 4, 3)`, x can be any interger
* filters  
    1 filters, size = `(1, 1)`, stride = `(1, 1)`
* activation funtion: `sigmoid` 

## Problems and Analysis
### Shuffle Before Training
In `keras` doc, it says the model would do the data shuffle automaticlly before training.  
But as the labeled data was sorted by `off_edge, on_edge, on_corner`, and the `size(training set) / size(test set) = 70 : 30`, if we just take first 70% of data as training data will result in most of `on_corner` data being devided into `test set`, which would lead to extremly awful accuracy rate.  
So shuffle before deividing data into `test set` and `training set` is required.
### Result Analyse  
The edge detection model result is sytisfying, the acc in test set can reach `99%`, but the edge deteced can be a llittle bit wide and bluring.  
I think it could be improved by enlarged NN.  
The corner detection model result is not quite pleasant, the acc only reach `96%`, and there are conditions that a whole thin horizontal edge being recognized as a corner.  
Up to now I still don't have any idea about it. 













