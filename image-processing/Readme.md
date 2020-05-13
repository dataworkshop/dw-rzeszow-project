# Sudoku - Image preprocessing


## Prerequisites

You need to know your camera ID.

You can use v4l-utils


```
sudo apt-get install v4l-utils

v4l2-ctl –list-devices

```


## Installing


### Exit the current venv
```
(shiny_new_env)$ conda deactivate
```
### Spin up a new one
```
$ conda create -n env_sudoku python=3.7
```
### Activate it
```
$ conda activate env_sudoku
```
### Install from our fancy new file
```
(env_sudoku)$ pip install -r requirements.txt
```

You need to install OpenCV 
https://pypi.org/project/opencv-python/

```
pip install opencv-python
```

## Running 


### With a camera as the input

Remeber to use your webcam ID

```
python image-processing.py cam 1
```

### With a file as the input 
```
python image-processing.py file test.jpg

```

Use "ESC" key to continue.

## Input
### test.jpg as input
![Image description](https://i.ibb.co/2sYVq6F/test.jpg)


## Output
### Step 1
![Image description](https://i.ibb.co/WGPxMPq/frame-unsharp-mask.jpg)
### Step 2
![Image description](https://i.ibb.co/P1BWH52/frame-bitwise-not.jpg)
### Step 3
![Image description](https://i.ibb.co/6wxjC3g/frame-cutted-Thresh.png)



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
