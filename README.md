# rustytracker

Webcam based object tracking using rust and opencv.

This is a webcam-based object tracker and glyph recognizer using rust and opencv.  I have wanted to try this for awhile, 
but I wanted to do it in rust and see if performance was good enough for it to be a useful controller for my home theater.
After searching around a bit, I found the pi-to-potter project https://github.com/mamacker/pi_to_potter , which I based 
this project on.  Both projects use openCV for camera interaction, image processing, and the KNearest algorithm to do the image searching. 

I wanted to do this project in rust specifically as an exercise for myself, and I found that the opencv bindings, while
functional, are not particularly straightforward if you don't know the API well enough to translate from the C++/python
implementations out there.  A native rust library probably would have saved me a lot of time, despite not having the 
history that opencv does. 

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
[image-dir]/[glyph-dir].sh is the name of the shell script that is executed when the glyph with that name is detected.

For a directory structure as follows:
```
images/ 
 + circle.sh
 + circle/
 |  + circle1.png
 |  + circle2.png
 |  + circle3.png
 + square.sh
 + square/
 |  + square1.png
 |  + square2.png
 |  + square3.png
 ```

There are two glyphs, one called circle and one called square, each with three training images and a shell script to 
execute when the glyph is detected.

# Algorithm

Rustytracker isolates the item of interest and then uses the KNearest ml algorithm from openCV to detect glyphs.  
- Grab a frame from the webcam
- Convert frame to HSV color space
- Use HSV thresholding to isolate the color of interest
- Use opencv's contour detector to detect contours in the image
- Take the center of each contour as a point in a connected list of glpyh line segments
- Stop detecting points when a frame is detected with no contours or when the last n contours detected are within some small bounds (object not moving)
- Determine scale factor to fit points into sample image size
- Draw connected points as lines into blank sample image
- Submit sample image to KNearest algorithm for comparison
- Call shell script on successful detection

