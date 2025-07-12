# Bollywood Classification Project
## By Ananthu Nair

This project is a machine learning image classifier, which can accurately classify images of the 5 famous Bollywood actors & actresses: Amitabh Bachchan, Alia Bhatt, Shar Rukh Khan, Deepika Padukone, and Hrithik Roshan. 

### Step 1. Data Collection:
    Used batch image processor to automatically download large quantities of images from google images for each actor/actress

### Step 2. Data Cleaning:
    Used OpenCV to automatically process images and determine quality. This included verifying that facial features (most importantly two eyes) were present and that a person could be recognized with Haar cascades. Valid images were cropped and saved while invalid images were discarded during this process. For images with multiple people, manual verification was needed to identify the target.