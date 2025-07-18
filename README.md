# Bollywood Classification Project
## By Ananthu Nair

This project is a machine learning image classifier, which can accurately classify images of the 5 famous Bollywood actors & actresses: Amitabh Bachchan, Kareena Kapoor, Shah Rukh Khan, Deepika Padukone, and Hrithik Roshan. 

### Step 1. Data Collection:

Used batch image processor to automatically download large quantities of images from google images for each actor/actress

### Step 2. Data Cleaning:

Used OpenCV to automatically process images and determine quality. This included verifying that facial features (most importantly two eyes) were present and that a person could be recognized with Haar cascades. Valid images were cropped and saved while invalid images were discarded during this process. For images with multiple people, manual verification was needed to identify the target.

### Step 3. Feature Engineering

During this step, each image is transformed using wavelet transformation. This converts the images into a form that the model can digest and identify key features from. The raw, cropped image and the tranformed image are then combined into one final image that will be used to train the model.

### Step 4. Model and Hyperparameter Tuning
After the training data had been finalized, the model could be trained and optimized. The data was split into training and test data (80/20 split) and then run with a sample model and hyperparameters. To identify which would be the optimal model, GridSearchCV was used to automate permutations of models and hyperparameters. The resulting model was then saved to a file.

### Step 5. Server Hosting & Web Dev
After the model is complete and fully functional, I set up a python flask server with a classify image function that uses the saved model to run predictions. The utility file contains the functions that the server will use to carryout this process, for example, functions to crop the uploaded image, perform wavelet transformations, stack the images, and then run the prediction and get an output. These functions use base64 encoded strings to process the images. Next, I created a very simple UI in html along with javascript code to accept user uploaded images and handle the hiding/showing of certain elements depending on the prediction results.

### Step 6. Cloud Deployment