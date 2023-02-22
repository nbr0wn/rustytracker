use opencv::{Error, ml, types, imgcodecs, imgproc, core, prelude::*, videoio, Result};

#[cfg(feature="have_gui")]
use opencv::highgui;

use std::time::Instant;
use std::fs;
use std::collections::HashMap;
use std::process::Command;
use rand::Rng;
use clap::Parser;
use regex::Regex;


// This is the width and height of the model data.  Increaet it for
// more accuracy but more processing power required
const IMAGE_DIM:i32 = 20;

// Contours smaller than this will be ignored
const MIN_CONTOUR_AREA:f64 = 0.1;

// These are the high and low ranges for the inRange threshold function.
// Use the python tool in the tools directory to generate these values.
const HSV_MIN_RANGE : &'static [i32] = &[78,139,111];
const HSV_MAX_RANGE : &'static [i32] = &[114,255,255];

const FRAME_WIDTH : i32 = 640;
const FRAME_HEIGHT : i32 = 480;

// When checking for whether or not motion has stopped, this is 
// how many recent points to consider and how small the motion
// should be
const MOTION_STOPPED_WINDOW:usize = 10;
const MOTION_STOPPED_AREA:i32 = 100;

// When checking for whether or not motion has stopped, this is 
// how many milliseconds to consider
const MOTION_STOPPED_MS:u128 = 1000;

const SIMPLIFY_FACTOR : f64 = 1.0;

// Command line argument definitions
#[derive(Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Path to training image and script directory
   #[arg(short,long, default_value_t=("images".to_string()))]
   image_dir: String,

   /// Show tracking image windows
   #[cfg(feature="have_gui")]	
   #[arg(short,long, default_value_t = true)]
   show: bool,

   /// When training, write new images here
   #[arg(short='n',long, default_value_t=("new_images".to_string()))]
   new_image_dir: String,

   /// Training mode - save new images as they are detected
   #[arg(short,long, default_value_t = false)]
   train: bool,

   /// Use grayscale thresholding for detection
   #[arg(short,long, default_value_t = false)]
   grayscale: bool,

   /// Show fps
   #[arg(short,long, default_value_t = true)]
   fps: bool,
}

// Apply a scale factor to a point
fn scale_point( origin: &core::Point, pt: core::Point, scale: core::Point2f) -> core::Point {
	core::Point::new(
		((pt.x - origin.x) as f32 * scale.x) as i32,
		((pt.y - origin.y) as f32 * scale.y) as i32
	)
}

// Check a sample image against the trained KNearest model
fn check_glyph( knn: &core::Ptr<dyn KNearest>, keypoints : &core::Vector<core::Point>, 
	args: &Args ) -> Result<u32> {

	// Make sure we have enough points to make at least one line
	if keypoints.len() < 2 { 
		return Err(Error{code:0, message:"Not enough points".to_string()}); 
	}

	// Simplify the keypoint list
	let mut simplified : core::Vector<core::Point> = core::Vector::default();
	imgproc::approx_poly_dp(keypoints, &mut simplified, SIMPLIFY_FACTOR, false).unwrap();

	// Create a sample IMAGE_DIMxIMAGE_DIM image
	let zeros = Mat::zeros(IMAGE_DIM, IMAGE_DIM,core::CV_32F).unwrap();
	let mut directional : core::Mat = zeros.to_mat().unwrap();
	let mut dest_image : core::Mat = zeros.to_mat().unwrap();

	// Find the bounds of the keypoints
	// Start with a 1x1 bounding box and grow it
	let pt = simplified.get(0).unwrap();
	let mut bbox = core::Rect::new(pt.x as i32, pt.y as i32, 1, 1);
	for idx in 1..simplified.len() {
		let kp = simplified.get(idx).unwrap();
		update_bounds(&mut bbox, &kp, 9999);
	}

	// Figure out the scale factor to draw everything within bounds
	// into an IMAGE_DIM x IMAGE_DIM image.  Need to take some caer 
	// here because scaling in X and Y to match the image dimensions
	// means you can never have a straigh vertical or horizontal line.
	// as it will get stretched in both dimensions.

	// Default to scaling in both dimensions
	let mut x_scale = IMAGE_DIM as f32 / bbox.width as f32;
	let mut y_scale = IMAGE_DIM as f32 / bbox.height as f32;

	// If the glyph is much larger in X or Y then preserve aspect ratio
	if bbox.width < bbox.height / 2 || bbox.height < bbox.width / 2 {
		if x_scale < y_scale { y_scale = x_scale; }
		if y_scale < x_scale { x_scale = y_scale; }
	}

	let scale_factor : core::Point2f = core::Point2f::new( x_scale, y_scale );
	
	// Get the top left corner of the bounding box
	let origin = core::Point::new( bbox.x, bbox.y);

	let mut color_val = 1.0;
	let color_delta = 0.9 / (simplified.len() - 1) as f64;

	// Draw line segments between keypoints into the image 
	let mut pt1 = scale_point(&origin, simplified.get(0).unwrap(), scale_factor);
	for keypt in simplified.iter().skip(1) {
		let pt2 = scale_point(&origin, keypt, scale_factor);
		
		imgproc::line(&mut directional, pt1, pt2, 
			core::VecN([color_val, 0.0, 0.0, 0.0]),
			2, 0, 0).unwrap();

		imgproc::line(&mut dest_image, pt1, pt2, 
			core::VecN([1.0, 0.0, 0.0, 0.0]),
			2, 0, 0).unwrap();

		pt1 = pt2;
		color_val -= color_delta;
	}

	// Pass the image into the knn classifier
	let mut result_idx = Mat::from_slice(&[1]).unwrap();
	let mut neighbors = Mat::default();
	let mut dist = Mat::default();
	let mut sample = img_to_sample(&dest_image);

	knn.find_nearest(&mut sample, 1, &mut result_idx, &mut neighbors, &mut dist).unwrap();

	if args.train {
		// Write the training image to disk
		let flags : core::Vector<i32> = core::Vector::default();
		let mut rng = rand::thread_rng();
		let val = rng.gen::<u32>();
		imgcodecs::imwrite(format!("{}/{}.png",args.new_image_dir, val).as_str(), 
			&dest_image, &flags).unwrap();
	}

	let res: f32 = *result_idx.at(0).unwrap();

	#[cfg(feature="have_gui")]	
	if args.show {
		// Draw the glyph 
		highgui::imshow("sample", &dest_image).unwrap();
	}

	Ok(res as u32)
}

// Convert an image to the proper format for our model
fn img_to_sample(img: &Mat) -> opencv::prelude::Mat  {
	let mut output = core::Mat::default();
	img.convert_to(&mut output, core::CV_32F, 1.0, 0.0).unwrap();

	output.reshape(0, 1).unwrap()
}

// Train our KNN model with images in the image directory.  Use the 
// directory names 
fn build_model( args: &Args ) -> (core::Ptr<dyn KNearest>, HashMap<u32,String>) {

	// Our training data and labels
	let mut sample_set = core::Mat::default();
	let mut label_set = core::Mat::default();

	let mut label_hash: HashMap<u32,String> = HashMap::new();

	// Create the KNearest model
	let mut knn = <dyn ml::KNearest>::create().unwrap();

	let mut glyph_index = 0;

	// Iterate over the glyph image directories in sorted order
	let training_paths : Vec<_> = fs::read_dir(args.image_dir.clone()).unwrap().map( 
		|dir| dir.unwrap()).collect();

	let re = Regex::new(r".*/").unwrap();
	// Iterate over the directories
	for glyph_dir in training_paths {
		let glyph_path = glyph_dir.path();
		if glyph_path.is_dir() {
		
			// Add the label to our glpyh id hash
			let path_str : &str = glyph_path.to_str().unwrap();
			let glyph_name = re.replace(path_str, "".to_string()).into_owned();
			println!("GLYPH INDEX {} is {:?}", glyph_index, glyph_name);
			label_hash.insert(glyph_index, glyph_name);
			// Loop through the images in the directory for this glyph
			for glyph_file in fs::read_dir(glyph_path).unwrap() {
				let labelidx = Mat::from_slice(&[glyph_index as f32]).unwrap();
				let glyph_file = glyph_file.unwrap();
				let img = imgcodecs::imread(glyph_file.path().to_str().unwrap(),
					imgcodecs::IMREAD_GRAYSCALE).unwrap();
				let sample = img_to_sample(&img);
				sample_set.push_back(&sample).unwrap();
				label_set.push_back(&labelidx).unwrap();
			}

			glyph_index += 1;
		}
	}

	if glyph_index > 0 {
	// Train the model with our sample set
	knn.train( &sample_set, ml::ROW_SAMPLE, &label_set ).unwrap();
	}

	// Return the goodies
	(knn,label_hash)
}

// Given a bounding box, update its bounds to include the new point.  Discard any
// new point that is within a threshold of the center
fn update_bounds( bounds: &mut core::Rect, pt: &core::Point, threshold: i32 ) {

	// First off make sure the new point is within threshold distance of the current center
	if (pt.x - (bounds.x + bounds.width / 2) > threshold)
	|| (pt.y - (bounds.y + bounds.height / 2) > threshold)  {
		// Too far - don't update
		return;
	}

	// Now adjust the bounding box to include the new point
	if pt.x < bounds.x {
		let diff = bounds.x - pt.x;
		bounds.x = pt.x ;
		bounds.width += diff;
	}

	if pt.y < bounds.y {
		let diff = bounds.y - pt.y;
		bounds.y = pt.y;
		bounds.height += diff;
	}

	if pt.x > bounds.x + bounds.width {
		bounds.width = pt.x - bounds.x;
	}

	if pt.y > bounds.y + bounds.height {
		bounds.height = pt.y - bounds.y;
	}
}


// Convenience function for center of a rectangle
fn get_center( rect: &core::Rect ) -> core::Point {
	core::Point { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2}
}

// Return the center point of the largest contour
fn get_max_contour_center( contours: &types::VectorOfMat ) -> Option<core::Point> {
	let mut max_area = MIN_CONTOUR_AREA;
	let mut found = false;
	let mut bounds: core::Rect = core::Rect::default();
	for c in contours.iter() {
		// TODO - maybe sufficient and faster to get the bounding rect first and check its area
		let a = imgproc::contour_area(&c,false).unwrap();
		if a > max_area {
			found = true;
			max_area = a;
			bounds = imgproc::bounding_rect(&c).unwrap();
		}
	}

	if found {
		return Some(get_center(&bounds));
	}

	None
}

#[cfg(feature="have_gui")]	
fn draw_glyph(dest_image: &mut core::Mat, points : &core::Vector<core::Point>) {

	// Don't bother if we don't have enough points
	if points.len() < 2 {
		return;
	}

	let mut simplified : core::Vector<core::Point> = core::Vector::default();
	imgproc::approx_poly_dp(points, &mut simplified, SIMPLIFY_FACTOR, false).unwrap();

	println!("Simplifying {} -> {} points", points.len(), simplified.len());

	// Draw the keypoints with connected line segments in decreasing 
	// intensity
	let mut prev_pt = simplified.get(0).unwrap();

	imgproc::draw_marker(dest_image, prev_pt, 
			core::VecN([0.0, 0.0, 255.0, 1.0]),
			imgproc::MARKER_DIAMOND,
			10,
			2,
			imgproc::FILLED).unwrap();

	let mut color_val = 255.0;
	let color_delta = 128.0 / (simplified.len() - 1) as f64;

	for keypt in simplified.iter().skip(1) {
		imgproc::draw_marker(dest_image, keypt, 
				core::VecN([0.0, 0.0, 255.0, 1.0]),
				imgproc::MARKER_DIAMOND,
				10,
				2,
				imgproc::FILLED).unwrap();
			
		imgproc::line(dest_image, prev_pt, keypt, 
			core::VecN([color_val, color_val, color_val, 1.0]), 2, 0, 0).unwrap();

		prev_pt = keypt;
		color_val -= color_delta;
	}

	// Get bounding box
	let pt = simplified.get(0).unwrap();
	let mut bbox = core::Rect::new(pt.x as i32, pt.y as i32, 1, 1);
	for idx in 1..simplified.len() {
		let kp = simplified.get(idx).unwrap();
		update_bounds(&mut bbox, &kp, 9999);
	}

	// Draw it
	imgproc::rectangle(dest_image, bbox,
		core::VecN([0.0, 255.0, 255.0, 1.0]), 2, 0, 0).unwrap();
}

fn main() -> Result<()> {

	let args = Args::parse();

	// It's annoying to have to do it this way
	#[cfg(feature="have_gui")]	
	let cam_win = "frame";
	#[cfg(feature="have_gui")]	
	let thresh_win = "threshold";
	#[cfg(feature="have_gui")]	
	let glyph_win = "glyph";
	#[cfg(feature="have_gui")]	
	let sample_win = "sample";

	#[cfg(feature="have_gui")]	
	highgui::named_window(cam_win, highgui::WINDOW_AUTOSIZE)?;
	#[cfg(feature="have_gui")]	
	highgui::named_window(thresh_win, highgui::WINDOW_AUTOSIZE)?;
	#[cfg(feature="have_gui")]	
	highgui::named_window(glyph_win, highgui::WINDOW_AUTOSIZE)?;
	#[cfg(feature="have_gui")]	
	highgui::named_window(sample_win, highgui::WINDOW_AUTOSIZE)?;
	#[cfg(feature="have_gui")]	
	let zeros = Mat::zeros(FRAME_WIDTH, FRAME_HEIGHT,core::CV_8UC3).unwrap();
	#[cfg(feature="have_gui")]	
	let mut glyph_image : core::Mat = zeros.to_mat().unwrap();


	// Build the model and get the glyph labels
	let (knn,labels) = build_model(&args);

	let mut all_keypoints : core::Vector<core::Point> = core::Vector::default();

	let mut quiet_start = Instant::now();

	// If we're training, create the new image dir
	if args.train {
		fs::create_dir_all(format!("{}",args.new_image_dir)).unwrap();
	}

	let mut frame = Mat::default();
	let mut gray = Mat::default();
	let mut thresh = Mat::default();
	let mut frame_count = 0;
	let mut frame_start = Instant::now();

	// HSV color range for threshold
	let mut hsv = Mat::default();
	let lower_bound = Mat::from_slice(&HSV_MIN_RANGE).unwrap();
	let upper_bound = Mat::from_slice(&HSV_MAX_RANGE).unwrap();

	loop {

		// Camera setup
		let mut cam = match videoio::VideoCapture::new(0, videoio::CAP_ANY) {
			Ok(cam) => cam,
			Err(e) => { 
				println!("Error Opening Camera: {}", e); 
				continue; 
			}
		};

		// Try again until we have a real camera
		if ! videoio::VideoCapture::is_opened(&cam)? {
			println!("Error Opening Camera");
			continue;
		}

		cam.set(videoio::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH as f64).unwrap();
		cam.set(videoio::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT as f64).unwrap();


		loop {
			let mut do_glyph_check: bool = false;

			if args.fps {
				if frame_count == 0 {
					frame_start = Instant::now();
				}
				frame_count += 1;
			}

			// Grab a frame from the camera
			cam.read(&mut frame)?;

			// If we didn't get a frame, try for another
			if frame.size()?.width <= 0 {
				// Unless the camera is fubar, in which case just break and
				// go back to the setup loop
				if ! videoio::VideoCapture::is_opened(&cam)? {
					println!("Camera Error.  Resetting");
					break;
				}
				continue;
			}

			if args.grayscale {
				// Convert to gray scale
				imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

				// Do binary thresholding on the gray image
				imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY).unwrap();
			} else {
				// Convert frame to hsv
				imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0).unwrap();

				// Do color thresholding
				core::in_range(&hsv, &lower_bound, &upper_bound, &mut thresh).unwrap();
			}

			// Blur the threshold image - sometimes this is helpful for noisy images
			//let mut intermediate = Mat::default();
			//imgproc::blur(&intermediate, &mut thresh, core::Size::new(7,7), core::Point::new(-1,-1), 0).unwrap();

			// Find contours in the thresholded image
			let mut found_contours = types::VectorOfMat::new(); 
			imgproc::find_contours(
				&mut thresh, 
				&mut found_contours, 
				imgproc::RETR_EXTERNAL,  // Only the external contours
				imgproc::CHAIN_APPROX_SIMPLE,  // Only the bounds are stored
				core::Point::new(0,0)
			)?;

			// Get the center of the largest contour if there is one
			if let Some(center_point) = get_max_contour_center(&found_contours) {
				// Add our keypoint to the list of keypoints
				all_keypoints.push(center_point);
				quiet_start = Instant::now();

			} else {
				// If we haven't seen a point for awhile, attempt a match
				if quiet_start.elapsed().as_millis() > MOTION_STOPPED_MS && all_keypoints.len() > 2 {
					println!("No More Points");
					do_glyph_check = true;
				}
			}

			// Get the bounds of the last few points and see if the source is not moving
			// No need to do this if we already decided that we need to do a glyph check
			if all_keypoints.len()  > 2 && !do_glyph_check {
				// We'll use the first point as our bounds start
				let mut skip: usize =1;

				// Figure out how points we need to skip
				if all_keypoints.len() > MOTION_STOPPED_WINDOW {
					skip = all_keypoints.len() - MOTION_STOPPED_WINDOW + 1;
				} 

				// Init our bounds with the first keypoint
				let kp = all_keypoints.get(skip-1).unwrap();
				let mut bounds = core::Rect::new(kp.x, kp.y, 1, 1);

				// Calculate bounding box for the points
				for (_idx, keypt) in all_keypoints.iter().skip(skip).enumerate() {
					update_bounds(&mut bounds, &keypt, 9999);
				}

				// Check area
				if all_keypoints.len() > 10 && bounds.area() < MOTION_STOPPED_AREA {
					println!("Area: {}", bounds.area());
					println!("Not enough motion");
					do_glyph_check = true;
				}
			}

			// Show our windows if requested
			#[cfg(feature="have_gui")]	
			if args.show {
				glyph_image = zeros.to_mat().unwrap();
				draw_glyph(&mut glyph_image, &all_keypoints);

				highgui::imshow(cam_win, &frame).unwrap();
				highgui::imshow(thresh_win, &thresh).unwrap();
				highgui::imshow(glyph_win, &glyph_image).unwrap();

				let key = highgui::wait_key(1).unwrap();
				if key > 0 && key != 255 {
					return Ok(());
				}
			}

			// Time to run our glyph through the model?
			if do_glyph_check {
				// Stopped.  Check the image and clear the points
				if let Ok(result) = check_glyph(&knn, &all_keypoints, &args) {
					println!("Glyph: {:?}", labels[&result]);
					let _ = Command::new(format!("{}/glyph.sh",args.image_dir))
					.arg(format!("{}",labels[&result]))
					.output();
				}
				all_keypoints.clear();
			}

			// Handle fps 
			if args.fps {
				if frame_count == 100 {
					println!("ms per frame:{}",frame_start.elapsed().as_millis() / 100);
					frame_count = 0;
				}
			}
		}
	}
}