use opencv::{highgui, Error, ml, types, imgcodecs, imgproc, core, features2d, prelude::*, videoio, Result};
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
const MIN_CONTOUR_AREA:f64 = 3.0;

// These are the high and low ranges for the inRange threshold function.
// Use the python tool in the tools directory to generate these values.
const HSV_MIN_RANGE : &'static [i32] = &[78,139,111];
const HSV_MAX_RANGE : &'static [i32] = &[114,255,255];

#[derive(Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Path to training image and script directory
   #[arg(short,long, default_value_t=("images".to_string()))]
   image_dir: String,

   /// Show tracking image windows
   #[arg(short,long, default_value_t = false)]
   show: bool,

   /// When training, write new images here
   #[arg(short='n',long, default_value_t=("new_images".to_string()))]
   new_image_dir: String,

   /// Training mode - save new images as they are detected
   #[arg(short,long, default_value_t = false)]
   train: bool,
}

fn scale_point( origin: &core::Point2f, pt: core::Point2f, scale: core::Point2f) -> core::Point2i {
	let pt = core::Point2i::new(
		((pt.x - origin.x) * scale.x) as i32,
		((pt.y - origin.y) * scale.y) as i32
	);

	pt
}

fn check_glyph( knn: &core::Ptr<dyn KNearest>, keypoints : &core::Vector<core::KeyPoint>, 
	args: &Args ) -> Result<u32> {

	// Make sure we have enough points to make at least one line
	if keypoints.len() < 2 { 
		return Err(Error{code:0, message:"Not enough points".to_string()}); 
	}

	// Create a 32x32 image - man these rust opencv bindings are annoying
	// Maybe there's a better way to do this but I couldn't find it
	let zeros = Mat::zeros(IMAGE_DIM, IMAGE_DIM,core::CV_32F).unwrap();
	let mut dest_image : core::Mat = zeros.to_mat().unwrap();

	// Find the bounds of the keypoints
	// Start with a 1x1 bounding box 
	let pt = keypoints.get(0).unwrap().pt();
	let mut bbox = core::Rect::new(pt.x as i32, pt.y as i32, 1, 1);
	for idx in 1..keypoints.len() {
		let kp = keypoints.get(idx).unwrap().pt();
		update_bounds(&mut bbox, kp.x as i32, kp.y as i32, 9999);
	}

	// Figure out the scale factor to draw everything within bounds
	// into an IMAGE_DIM x IMAGE_DIM image.  Need to take some caer 
	// here because filling the bounds means you can never have a 
	// straigh vertical or horizontal line.
	let mut x_scale = IMAGE_DIM as f32 / bbox.width as f32;
	let mut y_scale = IMAGE_DIM as f32 / bbox.height as f32;
	if x_scale < y_scale { y_scale = x_scale; }
	if y_scale < x_scale { x_scale = y_scale; }
	let scale_factor : core::Point2f = core::Point2f::new( x_scale, y_scale );
	
	// Get the top left corner of the bounding box
	let origin : core::Point2f = core::Point2f::new( bbox.x as f32, bbox.y as f32 );

	// Draw line segments between keypoints into the image 
	let mut pt1 = scale_point(&origin, keypoints.get(0)?.pt(), scale_factor);
	for keypt in keypoints.iter().skip(1) {
		let pt2 = scale_point(&origin, keypt.pt(), scale_factor);
		
		imgproc::line(&mut dest_image, pt1, pt2, 
			core::VecN([255.0, 255.0, 255.0, 1.0]), 2, 0, 0).unwrap();

		pt1 = pt2;
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

	if args.show {
		// Draw the glyph 
		let window = "glyph";
		highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
		highgui::imshow(window, &dest_image).unwrap();
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
	let mut training_paths : Vec<_> = fs::read_dir(args.image_dir.clone()).unwrap().map( 
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

	// Train the model with our sample set
	knn.train( &sample_set, ml::ROW_SAMPLE, &label_set ).unwrap();

	// Return the goodies
	(knn,label_hash)
}

// Given a bounding box, update its bounds to include the new point.  Discard any
// new point that is within a threshold of the center
fn update_bounds( bounds: &mut core::Rect, pt_x: i32, pt_y: i32, threshold: i32 ) {

	// First off make sure the new point is within threshold distance of the current center
	if (pt_x - (bounds.x + bounds.width / 2) > threshold)
	|| (pt_y - (bounds.y + bounds.height / 2) > threshold)  {
		// Too far - don't update
		return;
	}

	// Now adjust the bounding box to include the new point
	if pt_x < bounds.x {
		let diff = bounds.x - pt_x;
		bounds.x = pt_x ;
		bounds.width += diff;
	}

	if pt_y < bounds.y {
		let diff = bounds.y - pt_y;
		bounds.y = pt_y;
		bounds.height += diff;
	}

	if pt_x > bounds.x + bounds.width {
		bounds.width = pt_x - bounds.x;
	}

	if pt_y > bounds.y + bounds.height {
		bounds.height = pt_y - bounds.y;
	}
}

fn main() -> Result<()> {

	let args = Args::parse();

	let window = "video capture";
	highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

	let mut cam = match videoio::VideoCapture::new(0, videoio::CAP_ANY) {
		Ok(cam) => cam,
		Err(e) => { println!("Error Opening Camera: {}", e); return Err(e);  }
	};

	if let Err(e) = videoio::VideoCapture::is_opened(&cam) {
		println!("Error Opening Camera{}", e);
		return Err(e);
	}

    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 320.0).unwrap();
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 240.0).unwrap();


	// Build the model and get the glyph labels
	let (knn,labels) = build_model(&args);

	let mut all_keypoints : core::Vector<core::KeyPoint> = core::Vector::default();

	let mut quiet_start = Instant::now();

	// If we're training, create the new image dir
	if args.train {
		fs::create_dir_all(format!("{}",args.new_image_dir)).unwrap();
	}

	loop {
		//let frame_start = Instant::now();

		let mut frame = Mat::default();
		let mut hsv = Mat::default();
		let mut thresh = Mat::default();
		let mut dest_image = Mat::default();

		// HSV color range for threshold
		let lower_bound = Mat::from_slice(&HSV_MIN_RANGE).unwrap();
		let upper_bound = Mat::from_slice(&HSV_MAX_RANGE).unwrap();

		// Grab a frame from the camera
		cam.read(&mut frame)?;

		// Process it if it's real
		if frame.size()?.width > 0 {

			// Convert frame to hsv
			imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0).unwrap();

			// Do color thresholding
			core::in_range(&hsv, &lower_bound, &upper_bound, &mut thresh).unwrap();

			// Blur the threshold image
			//let mut intermediate = Mat::default();
			//imgproc::blur(&intermediate, &mut thresh, core::Size::new(7,7), core::Point::new(-1,-1), 0).unwrap();

			// Find contours in the thresholded image
			let zero_offset = core::Point::new(0, 0);
			let mut found_contours = types::VectorOfMat::new(); 
			imgproc::find_contours(
				&mut thresh, 
				&mut found_contours, 
				3, 
				1, 
				zero_offset
			)?;

			// Get rid of any contours that are too small
			let contours : types::VectorOfMat = found_contours.iter().take_while(
				|c| imgproc::contour_area(&c, false).unwrap() > MIN_CONTOUR_AREA).collect();


			if contours.len() > 0 {
				quiet_start = Instant::now();

				// Build the initial bounding box from the first contour
				let mut bounds = imgproc::bounding_rect(&contours.get(0)?)?;

				// Update the bounding box with the points from the rest of the contours
				for c in 1..contours.len() {
					let bbox = imgproc::bounding_rect(&contours.get(c)?)?;
					update_bounds( &mut bounds, bbox.x,            bbox.y,             30);
					update_bounds( &mut bounds, bbox.x+bbox.width, bbox.y+bbox.height, 30);
				}

				// Build the center point from our contour bounding box
				let pt = core::Point2f::new((bounds.x + bounds.width / 2) as f32, (bounds.y + bounds.height / 2) as f32);

				// Make a keypoint from the center point
				let kp = core::KeyPoint::new_point(pt,
					10.0,
					0.0,
					0.0,
					0,
					0 )?;

				// Add our keypoint to the list of keypoints
				all_keypoints.push(kp);

			} else {
				// remove all points if no contours are detected for awhile
				if quiet_start.elapsed().as_millis() > 500 {
					// Done drawing.  Check the image and clear the points.
					if let Ok(result) = check_glyph(&knn, &all_keypoints, &args) {
						println!("Glyph: {:?}", labels[&result]);
						let _ = Command::new(format!("{}/{}.sh",args.image_dir,labels[&result]))
						.output();
					}
					all_keypoints.clear();
				}
			}

			// Get the bounds of the last few blobs and see if the source is not moving
			let check_range:usize = 10;
			if all_keypoints.len()  > check_range {
				let mut min_x : f32 = 9999.0;
				let mut min_y : f32 = 9999.0;
				let mut max_x : f32 = 0.0;
				let mut max_y : f32 = 0.0;
				for (_idx, keypt) in all_keypoints.iter().skip(all_keypoints.len()-check_range).enumerate() {
					let pt = keypt.pt();
					if pt.x < min_x {
						min_x = pt.x;
					}
					if pt.y < min_y {
						min_y = pt.y;
					}
					if pt.x > max_x {
						max_x = pt.x;
					}
					if pt.y > max_y {
						max_y = pt.y;
					}
				}
				let area = (max_x - min_x) * (max_y - min_y);
				if area < 100.0 {
					// Stopped.  Check the image and clear the points
					if let Ok(result) = check_glyph(&knn, &all_keypoints, &args) {
						println!("Glyph: {:?}", labels[&result]);
						let _ = Command::new(format!("{}/{}.sh",args.image_dir,labels[&result]))
						.output();
					}
					all_keypoints.clear();
				}
			}
            
			// Plot the blobs
			let _ = features2d::draw_keypoints(&thresh, &all_keypoints, &mut dest_image, 
                core::VecN([0.0, 0.0, 255.0, 0.0]),
				features2d::DrawMatchesFlags::DEFAULT);

			// Draw line segments between keypoints
			if all_keypoints.len() > 2 {
				let mut pt1 = all_keypoints.get(0)?.pt();
				for keypt in all_keypoints.iter().skip(1) {
					let pt2 = keypt.pt();
					let pt2_u32 = core::Point2i {x:pt2.x as i32, y:pt2.y as i32};
					let pt1_u32 = core::Point2i {x:pt1.x as i32, y:pt1.y as i32};
					imgproc::line(&mut dest_image, pt1_u32, pt2_u32, 
						core::VecN([0.0, 255.0, 255.0, 0.0]), 2, 0, 0)?;

					pt1 = pt2;
				}

			let pt = all_keypoints.get(0).unwrap().pt();
			let mut bbox = core::Rect::new(pt.x as i32, pt.y as i32, 1, 1);
			for idx in 1..all_keypoints.len() {
				let kp = all_keypoints.get(idx).unwrap().pt();
				update_bounds(&mut bbox, kp.x as i32, kp.y as i32, 9999);
			}
			imgproc::rectangle(&mut dest_image, bbox,
				core::VecN([0.0, 255.0, 255.0, 0.0]), 2, 0, 0)?;
			}

			if args.show {
				highgui::imshow(window, &dest_image)?;
			}
		}
		if args.show {
			let key = highgui::wait_key(100)?;
			if key > 0 && key != 255 {
				break;
			}
		}
		//println!("F:{}",frame_start.elapsed().as_millis());
	}
	Ok(())
}