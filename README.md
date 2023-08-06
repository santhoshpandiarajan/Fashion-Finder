# Fashion-Finder Using OneApi


## Style Match Made Easy Where Fashion Meets AI!

Welcome to the Fashion Product Image Similarity project, a demonstration of how machine learning techniques can be harnessed to identify and categorize visually similar fashion products. In this project, we aim to explore the capabilities of transfer learning and K-Nearest Neighbors (KNN) classification to build an efficient and accurate fashion image similarity system.

The objective of this project is to create an intelligent solution that aids fashion retailers and e-commerce platforms in providing personalized product recommendations to their customers. By leveraging a pre-trained VGG16 convolutional neural network, we extract essential features from fashion product images, transforming them into meaningful embeddings. These embeddings represent the distinctive characteristics of each product image.

The process involves training a KNN classifier on the generated embeddings to establish relationships between products based on visual similarities. This allows us to efficiently find the nearest neighbors to a given image, enabling quick and effective similarity searches.

The Fashion Finder project leverages the power of Intel oneAPI and the Intel Extension for Scikit-learn (sklearnex) to enhance the performance and capabilities of the machine learning algorithms used for image similarity and product recommendation.

Intel oneAPI provides a comprehensive software toolkit that enables developers to optimize code for various hardware architectures, including CPUs and GPUs. With oneAPI, the algorithms in the Fashion Finder project can be executed on different devices, such as CPUs and GPUs, to take advantage of the parallel processing capabilities of GPUs, resulting in faster computations and improved performance.

The Intel Extension for Scikit-learn (sklearnex) further extends the functionality of the scikit-learn library by integrating oneAPI concepts and the dpctl package. This integration allows the algorithms to be offloaded to specific devices using dpctl.tensor.usm_ndarray for input data or by setting global configurations using options like target_offload and allow_fallback_to_host.


In the subsequent sections, we will delve into the details of the project, starting with data preparation and image preprocessing. We will explore how the VGG16 model is utilized for feature extraction, and how KNN classification facilitates image similarity analysis.

This project is a valuable showcase of how modern machine learning techniques can be applied to real-world scenarios, enhancing the functionality of fashion-related applications and contributing to a more personalized shopping experience.

![1](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/bedef93e-d93c-45d9-b517-a77718754869)

In this project, we utilize several essential Python packages for building an AI-powered fashion product image similarity system. 

1. **NumPy**: NumPy is the fundamental package for scientific computing with Python. It provides powerful tools for working with arrays and matrices, allowing us to efficiently handle and manipulate numerical data.

2. **Pandas**: Pandas is a widely-used data manipulation library. It helps us manage and analyze structured data in tabular form, making it easier to read, process, and prepare datasets for training and analysis.

3. **TensorFlow**: TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive set of tools and functionalities for building and training deep learning models, including neural networks, making it an essential package for our image similarity project.

4. **Matplotlib**: Matplotlib is a plotting library that enables us to create various types of visualizations. It helps us visualize data, model performance, and image similarity results, making it easier to understand and communicate our findings.

5. **Seaborn**: Seaborn is a data visualization library built on top of Matplotlib. It provides a higher-level interface for creating attractive and informative statistical graphics, enhancing the aesthetics of our plots.

6. **tensorflow.keras.layers and tensorflow.keras.models**: These modules are part of TensorFlow's high-level Keras API, which offers a user-friendly way to define neural network architectures and models. We use these modules to build the image similarity model using VGG16, a pre-trained deep learning model for image feature extraction.

7. **tensorflow.keras.preprocessing.image**: This module contains utilities for loading and preprocessing images for deep learning models. It allows us to efficiently prepare the fashion product images for input to our image similarity model.

Overall, these packages play crucial roles in the development and implementation of our AI-driven fashion product image similarity system, providing essential functionalities for data manipulation, deep learning, and data visualization.

![2](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/098cf2f7-f4cb-4e21-bbbb-de3d68c4e0bf)

In these lines of code, we perform data preparation and load the necessary datasets for the fashion product image similarity project. We first set the default font size for all matplotlib plots to 16 using `plt.rcParams['font.size'] = 16`. Next, we define the `path` variable to specify the location where the fashion product images are stored. Then, we read the "images.csv" file using `pd.read_csv()` and store the data in the `images_df` DataFrame. This DataFrame will hold metadata related to each fashion product image, such as image names and corresponding labels. Similarly, we read the "styles.csv" file using `pd.read_csv()` and store the data in the `styles_df` DataFrame. This DataFrame will contain information about different fashion styles, including style IDs and corresponding image filenames. Please note that the `on_bad_lines` parameter in the `pd.read_csv()` function is used to handle lines with too many fields, and it can be adjusted based on the dataset's formatting.

![3](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/ecc13ae4-b53b-41d4-b25c-46916e734c38)
![4](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/9f526bda-6ad4-4853-89d3-50f289f564b6)
![5](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/30824453-02e3-4c8c-8562-d91c09392958)
![6](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/c0334701-d244-4a36-a621-bd4037624177)

In this code snippet, we perform data exploration and preprocessing on the `images_df` and `styles_df` DataFrames. We first use `images_df.head()` to view the initial rows of the `images_df` DataFrame, which contains metadata about the fashion product images. Similarly, we use `styles_df.head()` to inspect the first few rows of the `styles_df` DataFrame, which contains information about different fashion styles, including style IDs. To facilitate further analysis, we create a new column called "filename" in the `styles_df` DataFrame using `styles_df['filename'] = styles_df['id'].astype('str')+'.jpg'`. This column will store the image filenames corresponding to each style, allowing us to link styles with their respective images. Lastly, we display the updated `styles_df` DataFrame using `styles_df.head()` to verify that the "filename" column has been added successfully. These initial data exploration and preprocessing steps are crucial for understanding the structure of the datasets and preparing the data for subsequent image processing and similarity analysis.

![7](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/d7aa6763-673c-4499-bfcc-2f84e3ebcb4c)

In this section of the code, we perform additional data processing to ensure that the fashion product images mentioned in the "styles.csv" DataFrame are present in the specified image directory. First, we use the `os.listdir()` function to obtain a list of all image files present in the directory specified by the `path` variable. This step collects the actual image filenames available in the image directory. Next, we create a new column named "present" in the `styles_df` DataFrame. Using the `apply()` function with a lambda function, we check if each image filename (stored in the "filename" column) from the `styles_df` DataFrame exists in the `image_files` list. The result of this operation is a boolean value indicating whether each image is present in the image directory. 

![8](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/c316f68e-34dd-418a-aa31-4be94b2ef410)

Finally, we filter the `styles_df` DataFrame to keep only the rows where the corresponding image files are available (i.e., where the "present" column is True). We then reset the DataFrame's index to ensure continuity in the row numbering. The purpose of this data processing is to exclude any style entries from the `styles_df` DataFrame that do not have associated images in the image directory. This step ensures that the data used for image similarity analysis is consistent and complete, containing only the styles for which we have available images.

![10](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/d3edf6a2-2e41-491f-96bf-83170d36a109)

In this code segment, we set up an ImageDataGenerator to preprocess fashion product images and create a generator for loading these images from a specified image directory. The ImageDataGenerator is a TensorFlow utility that performs real-time data augmentation and image preprocessing.

We define `img_size = 224` to set the target image size for preprocessing. All images will be resized to a square of 224x224 pixels, which is a common size used for deep learning models. Using `ImageDataGenerator(rescale=1./255)`, we create a data generator and rescale the pixel values of the images to be in the range [0, 1] by dividing all pixel values by 255. This ensures the pixel values are suitable for input to deep learning models.

The generator is then created with `datagen.flow_from_dataframe()`, which directly loads image data from the `styles_df` DataFrame and applies the specified preprocessing steps. It uses `styles_df` to obtain information about the images, including their filenames and any associated metadata.

The generator uses the specified image directory (`path`) to locate the fashion product images. It also resizes the images to the target size (224x224), reads image filenames from the 'filename' column of `styles_df`, and sets the batch size to 32.

Since we are not performing classification in this context, `class_mode` is set to None. The generator will not expect class labels for the images. Finally, `shuffle` is set to False, meaning the generator will not shuffle the images during data loading, and `classes` is set to None since we are not performing classification.

![11](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/654d99e1-b595-42d3-811e-4c37b99cc96e)
![12](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/852a304d-5fb1-482d-929a-fefd4a988267)

In this code segment, we utilize the VGG16 model, a pre-trained deep learning model from TensorFlow's Keras library, to create an image feature extraction model. This model is used to extract embeddings, which are informative and condensed representations of the images.

First, we load the VGG16 model and exclude its top classification layers, retaining only the feature extraction layers. The input shape of the images is specified as (img_size, img_size, 3) to accommodate images of size `img_size x img_size` with three color channels (RGB).

Next, we freeze the weights of all layers in the base model by setting them as non-trainable. This ensures that the pre-trained weights remain unchanged, and only the subsequently added layers will be trainable during model training.

We then create an input layer for the new image feature extraction model, specifying the shape of the input images. The input layer is passed through the base model to apply the VGG16 feature extraction layers to the input images. The resulting output represents the extracted features of the images. A global average pooling operation is applied to the feature maps to reduce their spatial dimensions to a fixed size. This operation aggregates the features of the images.

Finally, we create the embeddings model by defining its inputs as the input layer and its outputs as the global average pooling layer's output. This model represents the image feature extraction model, capable of producing embeddings for any input image. The summary of the embeddings model is then displayed, providing an overview of the model's architecture, layer types, output shapes, and the number of trainable parameters.

By leveraging the VGG16 model as a feature extractor and incorporating global average pooling, we efficiently obtain image embeddings. These embeddings serve as informative and compact representations of the images, suitable for various tasks, such as image similarity analysis or downstream applications.


![13](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/e4f3189b-4bde-4a9c-ae7d-8a9d96d653d5)

In this code segment, we use the `embeddings` model to generate image embeddings for a batch of fashion product images. The `predict()` method is employed with the `generator` object, which loads and preprocesses the fashion product images. The resulting embeddings represent informative and compact representations of the images and can be used for various tasks, such as image similarity analysis, clustering, or classification. By computing embeddings for the entire dataset, we create a feature representation suitable for further analysis and tasks related to fashion product images.

![14](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/0beda1e4-740d-415f-ad9d-a5023d97c140)

The `read_img` function is used to read and preprocess fashion product images from a specified image path. Its purpose is to prepare the images for further analysis or model training. The function takes the image path as input and performs the following steps sequentially.

First, it loads the image using the `load_img` function from the `tensorflow.keras.preprocessing.image` module. The `os.path.join()` method is used to combine the directory path (`path`) with the image filename (`image_path`), creating the complete path to the image. Additionally, the `target_size` parameter is set to `(img_size, img_size, 3)` to ensure that the image is resized to the desired target size with three color channels (RGB).

Next, the loaded image is converted to a NumPy array using the `img_to_array` function. This conversion is necessary as the VGG16 model, or other deep learning models, expect input data in NumPy array format.

Subsequently, the function scales the pixel values of the image to be in the range [0, 1]. This is achieved by dividing each pixel value by 255. Scaling the pixel values to this range is crucial for normalizing the data, ensuring that the model can efficiently process the images.

Finally, the preprocessed image, now in NumPy array format with scaled pixel values, is returned by the function. The image is now ready for use as input to the image feature extraction model or for any other image-related tasks. The preprocessing steps carried out by the `read_img` function ensure that the images are in a suitable format for effective analysis, classification, or other image-related tasks.

The line `styles_df = styles_df.reset_index(drop=True)` simply resets the index of the DataFrame `styles_df` to a default numerical index starting from 0, removing the previous index. This reorganization ensures a consistent and ordered indexing scheme for the DataFrame, which can be beneficial for data manipulation and analysis tasks.

![15](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/52da4c79-3c65-4c69-a30c-b1a92bf0bf3f)

This section of code starts by importing modules and packages, such as `random` and `sklearnex` to integrate Intel oneAPI optimizations. The `patch_sklearn()` function from `sklearnex` is used to seamlessly integrate oneAPI concepts with scikit-learn.

Next, the `KNeighborsClassifier` class is imported from `sklearnex.neighbors`, an extended version of scikit-learn's `KNeighborsClassifier`, which leverages Intel oneAPI optimizations for improved efficiency. The `styles_df` DataFrame contains the target labels ('id') used for training the `KNeighborsClassifier`. A new instance of the classifier is created with `n_neighbors=7`, indicating the number of neighbors used for classification. The model is then trained using the `fit()` method, where the feature data `X` and the target labels `y` are provided.

With the integration of `sklearnex`, the algorithm is optimized to run efficiently on different hardware, including CPUs and GPUs, resulting in faster and more accurate image similarity calculations.

The usage of `sklearnex` in this project is essential as it taps into Intel oneAPI optimizations, improving the performance of the `KNeighborsClassifier`. By leveraging oneAPI concepts, the Fashion Finder project achieves faster and more accurate image similarity comparisons, enabling more efficient and personalized fashion product recommendations.

After training, the KNN classifier can be utilized to predict the class labels of new or unseen fashion product images based on their feature vectors. The KNN algorithm is particularly useful for image classification tasks and can be applied to various other types of data as well.

![16](https://github.com/santhoshpandiarajan/Fashion-Finder/assets/131739054/ae9253b0-c87b-4941-9cfc-8ce09c5b2727)

This code segment finds similar fashion products using the trained k-nearest neighbors (KNN) classifier. It does the following:

In a loop of ten iterations, it randomly selects a fashion product image from the DataFrame `styles_df`.

The selected image is preprocessed using the `read_img` function.

The KNN classifier is used to find the k-nearest neighbors of the selected image's feature vector.

The original fashion product image is displayed in a small plot with the title "Input Image," and axis ticks are removed.

Five subplots are created to display the five most similar fashion products found by the KNN classifier.

For each nearest neighbor, the respective fashion product image is preprocessed and displayed with the title "Similar Product #i," where `i` represents the neighbor's rank (1st, 2nd, 3rd, etc.). Axis ticks are removed from the subplots.

By running this code, you can visualize the input image and its five most similar fashion products.
