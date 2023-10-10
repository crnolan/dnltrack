# dnltrack
Behavioural tracking software for the UNSW Decision Neuroscience Lab

## DepthAI installation

```
conda create -n depthai -c conda-forge python=3.10 numpy av pyqt pillow jupyter ipykernel
conda activate depthai
pip install opencv-python opencv-contrib-python depthai
```

Alternative install mostly with pip

```
conda create -n depthai -c conda-forge python=3.11 pip
pip install numpy av pyqt5 pillow jupyter ipykernel opencv-python opencv-contrib-python depthai
```