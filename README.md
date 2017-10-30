Image Dehazing Project Description

The objective of the project is to improve the retinal image quality assessment accuracy for Diabetic Retinopathy (DR) detection. We are using color balanced images and focusing on haze removal using dark channel prior method.  In the end, we have a patch-level classifier that outputs if a given patch is healthy or pathological. An healthy image should only contain healthy patches while a pathological image should contain at least one pathological patch.
We found out that using color balanced images consistently improved the results. We think that happens because the images look more similar across the dataset. The color balanced images are more well exposed and the colors and illumination is more consistent across the dataset which is probably the reason of the improved results.
It is common to preprocess the images, but the idea was that maybe we can learn the preprocessing function from data. The first step would be to train a neural network to map from the eye fundus images to their color balanced version. Then, we use the output of this network as the input to our DR detection model. In the end, we can fine tune the color balancing model to output a better representation.
Ideally, we could explore developing a more generic model that, for instance, could try to learn a model that outputs images that look the same (minimize some distance metric to the mean of the normalized images) while preserving the structures (minimizing some distance to the input image). At the same time, it is outputting an image that is useful for DR detection, by being trained jointly with the DR detection model.

Run Anaconda:
/Volumes/Users/xinhez/anaconda2/bin/python2.7

Collaborators:
Saniya Karnik
Xinhe Zhang
