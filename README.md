# sight corrector
adjust the view direction during webcam meetings

In webmeetings, people usually look at the monitor, not at the wecam. This gives the talk partner the impression not being looked at. This software corrects the misalignment at the talk partner's side.
In a first step, a short video sequence is recorded during which the person looks into the webcam and moves the head sideways and up and down. This records the eyes and their surroundings and saves them as a reference for later use.

In the subsequent webmeeting, the eye area with the wrong sight direction is replaced by these reference eyes.

The result is displayed in a separate window and can be transmitted by screen sharing, e.g.


Installation is not straight forward yet. For me this works:

cd to your desired working directory
conda create -n test
conda activate test
conda install python=3.7.6
pip install opencv-python
pip install mediapipe
pip install matplotlib
pip install scipy

start the program:
python sico.py 

