# Facial-Detection
Python Script for Facial Landmark Detection using OpenCv and dlib

Facial Detection.py asks the user to input the video path adn generates frames
and saves into two folders:
<ul>
<li>Open - Images in which a person's mouth is open (speaking)</li>
<li>Close - Images in which a person's mouth is closed (not speaking)</li>
</ul>

Functions are explained in the comments of the script.

The Frames are classified as open mouth/close mouth depending on the distance 
between the lip coordinates. 
The distance between the coordinates is comapred with a threshold which is the basis 
for classifying the images.
