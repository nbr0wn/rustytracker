use opencv::{highgui, Error, ml, types, imgcodecs, imgproc, core, features2d, prelude::*, videoio, Result};
use std::time::Instant;
use std::fs;
use rand::Rng;

    /*
	https://blog.devgenius.io/rust-and-opencv-bb0467bf35ff
https://learnopencv.com/blob-detection-using-opencv-python-c/
	pub struct SimpleBlobDetector_Params {
		pub threshold_step: f32,
		pub min_threshold: f32,
		pub max_threshold: f32,
		pub min_repeatability: size_t,
		pub min_dist_between_blobs: f32,
		pub filter_by_color: bool,
		pub blob_color: u8,
		pub filter_by_area: bool,
		pub min_area: f32,
		pub max_area: f32,
		pub filter_by_circularity: bool,
		pub min_circularity: f32,
		pub max_circularity: f32,
		pub filter_by_inertia: bool,
		pub min_inertia_ratio: f32,
		pub max_inertia_ratio: f32,
		pub filter_by_convexity: bool,
		pub min_convexity: f32,
		pub max_convexity: f32,
	}
    */

fn scale_point( origin: &core::Point2f, pt: core::Point2f, scale: core::Point2f) -> core::Point2i {
	let pt = core::Point2i::new(
		((pt.x - origin.x) * scale.x) as i32,
		((pt.y - origin.y) * scale.y) as i32
	);

	pt
}
const IMAGE_DIM:i32 = 20;
const MIN_CONTOUR_AREA:f64 = 3.0;


fn check_glyph( knn: &core::Ptr<dyn KNearest>, keypoints : &core::Vector<core::KeyPoint> ) -> Result<()> {

	if keypoints.len() < 3 { return Err(Error{code:0, message:"Not enough points".to_string()}); }

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

	// Write the training image to disk
	let flags : core::Vector<i32> = core::Vector::default();
	let mut rng = rand::thread_rng();
	let val = rng.gen::<u32>();
	imgcodecs::imwrite(format!("train_images/{}.png",val).as_str(), &dest_image, &flags);

	let res: f32 = *result_idx.at(0).unwrap();
	println!("RESULT:{}", res);

	// Draw the glyph 
	let window = "glyph";
	highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
	highgui::imshow(window, &dest_image).unwrap();

	Ok(())
}

// Convert an image to the proper format for our model
fn img_to_sample(img: &Mat) -> opencv::prelude::Mat  {
	let mut output = core::Mat::default();
	img.convert_to(&mut output, core::CV_32F, 1.0, 0.0).unwrap();

	output.reshape(0, 1).unwrap()
}

fn build_model( ) -> core::Ptr<dyn KNearest> {

	// Our training data and labels
	let mut sample_set = core::Mat::default();
	let mut label_set = core::Mat::default();

	// Create the KNearest model
	let mut knn = <dyn ml::KNearest>::create().unwrap();

	let dir = "images";

	let mut glyph_index = 0;

	// Iterate over the glyph image directories
	for glyph_dir in fs::read_dir(dir).unwrap() {
		let glyph_dir = glyph_dir.unwrap();
		let glyph_path = glyph_dir.path();
		println!("GLYPH INDEX {} is {:?}", glyph_index, glyph_path);
		if glyph_path.is_dir() {
			// Loop through the images in the directory for this glyph
			for glyph_file in fs::read_dir(glyph_path).unwrap() {
				let labelidx = Mat::from_slice(&[glyph_index as f32]).unwrap();
				let glyph_file = glyph_file.unwrap();
				let img = imgcodecs::imread(glyph_file.path().to_str().unwrap(),imgcodecs::IMREAD_GRAYSCALE).unwrap();
				let sample = img_to_sample(&img);
				sample_set.push_back(&sample).unwrap();
				label_set.push_back(&labelidx).unwrap();
			}
		}
		glyph_index += 1;
	}

	// Train the model with our sample set
	knn.train( &sample_set, ml::ROW_SAMPLE, &label_set ).unwrap();

	knn
}

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
	let window = "video capture";
	highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

	let knn = build_model();

	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 320.0).unwrap();
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 240.0).unwrap();

	let opened = videoio::VideoCapture::is_opened(&cam)?;
	if !opened {
		panic!("Unable to open default camera!");
	}

	//let mut all_contours = types::VectorOfMat::new(); 

	let mut all_keypoints : core::Vector<core::KeyPoint> = core::Vector::default();

	let mut quiet_start = Instant::now();

	/* 
	let mut bd_params : features2d::SimpleBlobDetector_Params = 
		features2d::SimpleBlobDetector_Params::default().unwrap();
	//bd_params.threshold_step = 2.0;
	//bd_params.min_threshold = 200.0;
	//bd_params.max_threshold = 255.0;
	bd_params.min_dist_between_blobs = 100.0;
	bd_params.filter_by_color = true;
	bd_params.blob_color = 255;
	//bd_params.filter_by_inertia = false;
	bd_params.filter_by_circularity = false;
	//bd_params.min_circularity = 0.05;
	bd_params.filter_by_convexity = false;
	bd_params.filter_by_area = true;
	bd_params.min_area = 15.0;
	let mut bd  = features2d::SimpleBlobDetector::create(bd_params).unwrap();
	*/

//   let mut detector = core::SparsePyrLKOpticalFlow::create(
//	core::Size(21,21),
// 		3,
//	TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01),
//	0, 
//	1e-4);

	loop {
		//let frame_start = Instant::now();

		let mut frame = Mat::default();
		let mut hsv = Mat::default();
		let mut thresh = Mat::default();
		let mut dest_image = Mat::default();

		// HSV color range for threshold
		let lower_bound = Mat::from_slice(&[78,139,111]).unwrap();
		let upper_bound = Mat::from_slice(&[114,255,255]).unwrap();

		// Grab a frame from the camera
		cam.read(&mut frame)?;

		// Process it if it's real
		if frame.size()?.width > 0 {

			// Convert frame to hsv
			imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0).unwrap();

			// Do color thresholding
			core::in_range(&hsv, &lower_bound, &upper_bound, &mut thresh).unwrap();

			//let mut intermediate = Mat::default();
			// Blur the intermediate image
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

/*
				// Add our contour to the list of contours
				all_contours.extend(contours);

				let hierarchy = Mat::default();
				let offset = core::Point::new(0, 0);

				// Draw the contours
				for c in 0..all_contours.len() {
					imgproc::draw_contours(
						&mut frame, 
						&all_contours, 
						c as i32,
						core::VecN([0.0, 255.0, 255.0, 0.0]),
						-1,
						1,
						&hierarchy,
						0,
						offset
					).unwrap();
				}
*/
			} else {
				// remove all points if no contours are detected for awhile
				if quiet_start.elapsed().as_millis() > 500 {
					// Done drawing.  Check the image and clear the points.
					//all_contours.clear();
					let _result = check_glyph(&knn, &all_keypoints);
					all_keypoints.clear();
				}
			}

/*
            // Detect the blobs
            let mut keypoints : core::Vector<core::KeyPoint> = core::Vector::default();
			let mask = core::no_array();
            let _ = bd.detect(&thresh, &mut keypoints, &mask);

			// remove all points if no blobs are detected for awhile
			if keypoints.len() == 0 {
				if quiet_start.elapsed().as_millis() > 500 {
					// Done drawing.  Check the image and clear the points.
					all_keypoints.clear();
					all_contours.clear();
				}
			}
			else {
				quiet_start = Instant::now();
			}

			for pt in &keypoints {
				print!("{} ", pt.size())
			}
			println!("");

			//all_keypoints.extend(keypoints);

*/


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
					//all_contours.clear();
					let _result = check_glyph(&knn, &all_keypoints);
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

            // Render them
            //for keypt in keypoints {
                //println!("{},{}", keypt.pt().x, keypt.pt().y);
            //}
            //imgproc::rectangle(&frame,
                //core::Rect::from_points(core::Point::new(0,0),core::Point::new(50,50)),
                //core::VecN([255.0,0.0,0.0,0.0]),
                //-1,
                //imgproc::LINE_8,
                //0);
			highgui::imshow(window, &dest_image)?;
		}
		let key = highgui::wait_key(100)?;
		if key > 0 && key != 255 {
			break;
		}
		//println!("F:{}",frame_start.elapsed().as_millis());
	}
	Ok(())
}