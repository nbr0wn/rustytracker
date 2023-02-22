# rustytracker

Webcam based object tracking using rust and opencv.

This is a webcam-based object tracker and glyph recognizer using rust and opencv.  I have wanted to try this for awhile, 
but I wanted to do it in rust and see if performance was good enough for it to be a useful controller for my home theater.
After searching around a bit, I found the pi-to-potter project https://github.com/mamacker/pi_to_potter , which I based 
this project on.  Both projects use openCV for camera interaction, image processing, and the KNearest algorithm to do the image searching.  For rustytracker, I added directionality to 
the glyphs, so a clockwise circle is different from a counter clockwise circle.

I wanted to do this project in rust as an exercise for myself, and I found that the opencv bindings, while
functional, are not particularly straightforward if you are not already intimately 
familiar with the API.  A native rust library probably would have saved me a lot of time, despite not having the history that opencv does, but I managed in the end.

![rustytracker](https://user-images.githubusercontent.com/1176032/219705995-7db77c5a-8ac2-434e-b1e9-17fd9427d755.png)


# How to use

```
Usage: rustytracker [OPTIONS]

Options:
  -i, --image-dir <TRAINING_DIR>       Path to training image and script directory [default: images]
  -t, --train                          Training mode - write images to [new-image-dir]
  -s, --show                           Show tracking windows
  -n, --new-image-dir <NEW_IMAGE_DIR>  Training mode - write images to training_dir [default: training-dir/new_images]
  -h, --help                           Print help
  -V, --version                        Print version

```

Rustytracker loads all image inside [image-dir] on startup, using [image-dir]/[glyph-dir]/*.png as training images.

For a directory structure as follows:
```
images/ 
 + glyph.sh
 + circle/
 |  + circle1.png
 |  + circle2.png
 |  + circle3.png
 + square/
 |  + square1.png
 |  + square2.png
 |  + square3.png
 ```

There are two glyphs, one called circle and one called square, each with three training images.
[image-dir]/glyph.sh is the name of the shell script that is executed on detection with 
the name of the detected glyph as its only argument.

# Algorithm

Rustytracker isolates the item of interest and then uses the KNearest ml algorithm from openCV to detect glyphs.  
- Grab a frame from the webcam
- Convert frame to HSV color space
- Use HSV thresholding to isolate the color of interest
- Use opencv's contour detector to detect contours in the image
- Take the center of each contour as a point in a connected list of glpyh line segments
- Stop detecting points when a frame is detected with no contours or when the last n contours detected are within some small bounds (object not moving)
- Determine scale factor to fit points into sample image size
- Draw connected points as lines into blank sample image with intensity dropping from start to end
- Submit sample image to KNearest algorithm for comparison
- Call shell script on successful detection


# Building on a raspberry pi

- sudo apt-get update
- sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev libopencv-dev -y
- cargo build
- Go get a coffee.  Maybe three.
