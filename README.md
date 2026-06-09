\# Sketch-to-Image AI System



\## Project Description



This project uses the Google Quick Draw dataset to classify hand-drawn sketches into three categories:



\* Flower

\* Animal

\* Building



A Convolutional Neural Network (CNN) was trained and improved through multiple experiments including Batch Normalization, Dropout, Early Stopping, Learning Rate tuning and Data Augmentation.



In the final stage, the CNN classifier was integrated with a Diffusion-based image generation system. The system classifies a user sketch, selects an appropriate prompt and generates a colored image automatically.



\---



\## Dataset



Google Quick Draw Dataset



Classes used:



\* Flower

\* Cat

\* Dog

\* House

\* Castle



Grouped into:



\* Flower

\* Animal

\* Building



\---



\## Final Model



Architecture:



\* CNN

\* Batch Normalization

\* Global Average Pooling

\* Dropout 0.3



Performance:



\* Validation Accuracy: 97.33%

\* Test Accuracy: 96.78%

\* Test Loss: 0.1225



\---



\## Sprint 4 Pipeline



Sketch Input

→ Image Preprocessing

→ CNN Classification

→ Category Selection

→ Prompt Selection

→ DreamShaper Diffusion Model

→ Generated Image



\---



\## Main Files



\* train\_baseline.py

\* train\_improved.py

\* train\_dropout03.py

\* predict\_category.py

\* cnn\_diffusion\_pipeline.py



\---



\## Run



Classify a sketch:



python predict\_category.py



Run the complete sketch-to-image pipeline:



python cnn\_diffusion\_pipeline.py



\---



\## Author



Edanur Özkaplan

&#x20;

20242013005

