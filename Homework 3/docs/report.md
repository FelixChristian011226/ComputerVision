# Homework3-Scene Recognition with Bag of Words

## 1. Setting

### 1.1 Objective

> The goal of this project is to give you a basic introduction to image recognition.
> Specifically, we will examine the task of scene recognition starting with a very simple
> method, e.g., tiny images and nearest neighbor classification, and then move on to
> bags of quantized local features.

The dataset consists of images from 15 different categories. Each category contains multiple images, and the task is to classify these images into their respective categories.

### 1.2 Environment

- **OS**: Linux (Ubuntu 22.04)
- **Lib**: Python 3, scikit-learn, OpenCV, NumPy.



## 2. Algorithm

### 2.1 Tiny Image + KNN

The **Tiny Image + KNN** method works by resizing images to a small fixed size and flattening the pixel values into a vector. This vector is used as a feature representation for the image, which is then classified using the k-Nearest Neighbors algorithm. 

- **Steps**:  
  	1. Resize the images to a smaller size (e.g., 32x32 pixels).  
  	1. Flatten the resized image into a 1D vector of pixel intensities.  
  	1. Use KNN to classify the image based on the distance between feature vectors. 

This method is simple but may not capture high-level patterns or features in the image, as it relies purely on raw pixel intensities. 

### 2.2 Bag of SIFT + KNN 

The **Bag of SIFT + KNN** method involves several key steps: 

 	1. **SIFT Feature Extraction**: Key points are detected in each image, and descriptors are computed for those key points. 
 	2. **Clustering**: SIFT descriptors are clustered into a predefined number of clusters (vocabulary size). 
 	3. **Feature Representation**: Each image is represented by a histogram of the number of occurrences of each visual word (cluster center). 
 	4. **KNN Classification**: The image's histogram is compared to the histograms of training images, and the class with the most neighbors is assigned to the image. 

In my code, SIFT features are extracted and clustered into visual words using k-means clustering. The resulting vocabulary size is tested with different numbers of clusters (10, 30, 50, 70, 100).



##  3. Experiments

### 3.1 Results

The following table summarizes the classification accuracy for both the **Tiny Image + KNN** method and the **Bag of SIFT + KNN** method with different visual vocabulary sizes.

| **Method**            | **Vocabulary Size** | **Average Accuracy** |
| --------------------- | ------------------- | -------------------- |
| **Tiny Image + KNN**  | N/A                 | 0.2236               |
| **Bag of SIFT + KNN** | 10                  | 0.2624               |
| **Bag of SIFT + KNN** | 30                  | 0.3354               |
| **Bag of SIFT + KNN** | 50                  | 0.3550               |
| **Bag of SIFT + KNN** | 70                  | 0.3568               |
| **Bag of SIFT + KNN** | 100                 | 0.3578               |

### 3.2 Accuracy Details

Here are the detailed accuracy results for each category in the dataset.

#### Tiny Image + KNN Results

| **Category** | **Accuracy** |
| ------------ | ------------ |
| coast        | 0.4154       |
| forest       | 0.1053       |
| highway      | 0.5375       |
| insidecity   | 0.0673       |
| mountain     | 0.1606       |
| office       | 0.1826       |
| opencountry  | 0.3581       |
| street       | 0.5000       |
| suburb       | 0.3546       |
| tallbuilding | 0.1719       |
| bedroom      | 0.1638       |
| industrial   | 0.0900       |
| kitchen      | 0.0818       |
| livingroom   | 0.1376       |
| store        | 0.0279       |

#### Bag of SIFT + KNN Results (Different Vocabulary Sizes)

| **Category** | **Vocabulary Size = 10** | **Vocabulary Size = 30** | **Vocabulary Size = 50** | **Vocabulary Size = 70** | **Vocabulary Size = 100** |
| ------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------- |
| coast        | 0.3308                   | 0.3423                   | 0.3846                   | 0.3769                   | 0.3654                    |
| forest       | 0.7237                   | 0.7763                   | 0.8289                   | 0.8377                   | 0.8026                    |
| highway      | 0.1938                   | 0.2812                   | 0.2938                   | 0.3125                   | 0.3375                    |
| insidecity   | 0.1394                   | 0.2452                   | 0.2788                   | 0.3029                   | 0.2740                    |
| mountain     | 0.2080                   | 0.3723                   | 0.4562                   | 0.4051                   | 0.4307                    |
| office       | 0.2696                   | 0.2174                   | 0.3304                   | 0.3478                   | 0.3130                    |
| opencountry  | 0.1871                   | 0.2516                   | 0.1839                   | 0.2290                   | 0.1935                    |
| street       | 0.2812                   | 0.3229                   | 0.3073                   | 0.2812                   | 0.2448                    |
| suburb       | 0.5106                   | 0.7021                   | 0.7730                   | 0.7376                   | 0.8014                    |
| tallbuilding | 0.1758                   | 0.2852                   | 0.3086                   | 0.3242                   | 0.3438                    |
| bedroom      | 0.1379                   | 0.2241                   | 0.1121                   | 0.1638                   | 0.1466                    |
| industrial   | 0.1090                   | 0.1611                   | 0.1801                   | 0.2370                   | 0.1848                    |
| kitchen      | 0.1455                   | 0.2091                   | 0.1636                   | 0.1182                   | 0.1818                    |
| livingroom   | 0.2487                   | 0.2963                   | 0.2487                   | 0.2275                   | 0.2487                    |
| store        | 0.2744                   | 0.3442                   | 0.4744                   | 0.4512                   | 0.4977                    |

## 4. Conclusion

- The **Tiny Image + KNN** method results in lower accuracy (0.2236 on average) because it relies solely on raw pixel values, which do not capture the complex features and patterns of the images effectively.
- **Bag of SIFT + KNN** performs much better, with the accuracy improving as the vocabulary size increases. The highest accuracy was obtained with a vocabulary size of 100, yielding an average accuracy of 0.3578.
- **Increasing the visual vocabulary size** improves the model’s ability to classify images by capturing more detailed and distinct features from the SIFT descriptors.

In conclusion, **Bag of SIFT + KNN** is a more effective method for image classification compared to **Tiny Image + KNN**, especially when a larger vocabulary is used.

