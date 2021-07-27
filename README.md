# Sight corrector
Adjusts the eyes' view direction during webcam meetings

## Description
In webmeetings, people usually look at the monitor, not at the webcam. This gives the talk partner the impression not being looked at. This software corrects the misalignment at the talk partner's side.
In a first step, a short video sequence is recorded during which the user looks into the webcam while moving the head sideways and up and down. This step records "reference eyes" for later use. In the subsequent webmeeting, the eyes with the wrong sight direction are replaced by these "reference eyes".

![alt text](sight_direction.jpg)

The result is displayed in a separate window and can be transmitted by screen sharing.


## Installation
The software was tested under 64-bit Ubuntu 20.04 with Python 3.7.6. It is recommended to create a virtual environment:

```
download the project files into a separate directory
cd to this directory
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage

`python sico.py`


## Customisation

If desired you can edit global_vars.py. as follows:

```
TWEAK_DENS (default 0.05, range >0 ... <1) # This is the density of the grid of the "reference eyes" sideways and up and down.

MAP_TOL (default 0.2) # This is the tolerance up to which eye mapping should be performed. It is calculated as the euclidean distance between current head orientation and closest grid point.

DAMPING (default 5) # This variable is used to reduce flickering. The value represents positional changes between two time adjacent video frames.

BTOL (default 1.01) # Threshold values for the relative difference in brightness between "reference eyes" and the current eyes for which eyes should not be tweaked.

DEVICE_IN (default '/dev/video0') # The device name of the used webcam. For macOS use '0' instead of '/dev/video0'.

MODE ('demo' or 'live') # 'demo' provides tweaked and original video shared in one window and including labelling. 'live' only shows tweaked video.
```
