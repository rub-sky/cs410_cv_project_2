/* ===== Written By Ruben Piatnitsky =====

===Moving Pedestrian and Vehicle Detection and Tracking===

This project needs to develop a system that can automatically detect a moving pedestrian and vehicle from a traffic surveillance video. You can implement the system or download the code available online.

Below are some public benchmark for this task.

1. http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Walking.zip
2. http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Subway.zip
http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html#cascade-classifier

Grading policy:
a. Implement moving pedestrain and vehicle detection. (40 points)
b. Implement moving pedestrain and vehicle tracking. (30 points)
c. In-class presentation (0-10 points)
d. Project report (0-20 points)
*/

// -c -v ../video/video1.avi

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

const string OUTPUT_FILENAME = "result.avi";
const string OUTPUT_IMG_FILENAME = "result.png";
vector<int> compression_params; // Image output params
int nextId = 0;

const int MAX_VIS = 20;
const int MAX_AGE= 6;
const int MIN_VIS = 8;
const double MAX_VIS_TO_AGE_RATIO = 0.6;
const float OVERLAP_THRESH = .35;
// Algorithm constants
const string TRK_ALG_KCF = "KCF";
const string TRK_ALG_MF = "MEDIANFLOW";
const string TRK_ALG_TLD = "TLD";

const int BLUR_SZ = 2;
// Haar Cascade Car Constants
const int CAR_SZ = 35;
const double CAR_SCALE_FACTOR = 1.5;
const int CAR_MIN_NEIGH = 1;
const int CAR_FLAGS = 0;
const cv::Size CAR_H_SIZE = cv::Size(20, CAR_SZ);
// Pedestrian Haar Constants
const int PED_SZ = 80;
const double PED_SCALE_FACTOR = 1.75;
const int PED_MIN_NEIGH = 3;
const int PED_FLAGS = 0; 
const cv::Size PED_H_SIZE = cv::Size(10, PED_SZ);
// Params for TrackerMIL
TrackerMIL::Params tmfParams;


struct trackedObj {
	int id;
	KeyPoint centroid;
	Rect2d bnd_box;
	int age = 1;
	int totVisCount = 1;
	int consecInvCount = 0;
	Ptr<Tracker> trkr;
	Ptr<TrackerMIL> trMedFlow;

	// Default constructor
	trackedObj(int idNum, KeyPoint cent, Rect2d bBox, int ageNum, int tvc, int cic, string alg)
	{
		id = nextId;

		centroid = cent;
		bnd_box = bBox;
		age = ageNum;
		totVisCount = tvc;
		consecInvCount = cic;

		trkr = Tracker::create(alg); // Initializes the tracking algorithm for the object
		//tmfParams.pointsInGrid = 150;
		trMedFlow = TrackerMIL::createTracker(tmfParams);
	}

	// Secondary constructor
	trackedObj(int idNum, KeyPoint cent, Rect2d bBox, string alg)
	{
		id = nextId;
		
		centroid = cent;
		bnd_box = bBox;

		trkr = Tracker::create(alg);  // Initializes the tracking algorithm for the object

		trMedFlow = TrackerMIL::createTracker(tmfParams);
	}

	void initTracker(Mat frame) {
		//trMedFlow->init(frame, bnd_box);
		trkr->init(frame, bnd_box);
	}

	// Updates the bounding box with the existing centroid
	void updateBBox() {
		Point pt_tl(centroid.pt.x + centroid.size, centroid.pt.y + centroid.size);
		Point pt_br(centroid.pt.x - centroid.size, centroid.pt.y - centroid.size);
		bnd_box = Rect2d(pt_tl, pt_br);
	}

	// Updates the invisible and age count for object
	void currentlyNotVisible() {
		++consecInvCount;
		++age;
	}

	// Updates the total visibility count and age. Also resets the invisible count to 0.
	void currentlyVisible() {
		consecInvCount = 0;
		++totVisCount;
		++age;
	}

	// For updating the bounding box
	void updateTracker(Mat frame) {
		// TrackerMedian Flow
		//trMedFlow->update(frame, bnd_box);
		trkr->update(frame, bnd_box);

		//trkr->update(frame, bnd_box);  // update the tracker for the bounding box
		//Point2f pt(bnd_box.x-(bnd_box.width/2), bnd_box.y - (bnd_box.height / 2));
		Point2f pt(bnd_box.x + (bnd_box.width/2), bnd_box.y + (bnd_box.height / 2));
		centroid = KeyPoint(pt, bnd_box.width / 2, -1,0,0,-1);

		
	}
};

/** Function Headers */
int load_cascades();
Mat detectAndDisplay(Mat frame);
int use_video(char* filename);
int use_image(char* filename);
vector<Rect2d> compareDetectedToTracked(vector<Rect> objs, Mat frame);  // For Tracking
KeyPoint rectToKeyPoint(Rect r);
Rect2d rectToRect2d(Rect r);
void deleteLostTracks();
float rectOverLap(Rect r1, Rect r2);

/** Global variables */
String car1_cascade_name = "../haar_cascades/haarcascade_car_1.xml";
String car2_cascade_name = "../haar_cascades/haarcascade_car_2.xml";
String car3_cascade_name = "../haar_cascades/cars3.xml";
String ped1_cascade_name = "../haar_cascades/haarcascade_pedestrian.xml";
String ped2_cascade_name = "../haar_cascades/haarcascade_ped.xml";
String ped3_cascade_name = "../haar_cascades/haarcascade_fullbody.xml";

CascadeClassifier car1_cascade;
CascadeClassifier car2_cascade;
CascadeClassifier car3_cascade;
CascadeClassifier ped1_cascade;
CascadeClassifier ped2_cascade;
CascadeClassifier ped3_cascade;
string window_name = "Detection and Tracking";
bool is_car = true;
RNG rng(12345);
int frame_cnt = 0;

/* For tracker*/
std::string trackingAlg = "MEDIANFLOW"; // KCF, MEDIANFLOW or TLD
vector<trackedObj> tracked_objs;


// Blob Detector
// Setup SimpleBlobDetector parameters.
SimpleBlobDetector::Params params;
SimpleBlobDetector blobDetect;


/** @function main */
int main(int argc, const char** argv)
{
	// Check args
	if (argc < 3) {
		cout <<	" Usage: <car or pedestrian> <image or video> <file_name>\n"
			 << "        <-c or -p>          <-i or -v>       <file_name>\n"
			 << " examples:\n"
			 << " example -c -i ./img/cars.jpg\n"
			 << " example -p -v ./ped.avi\n" << endl;
		cv::waitKey(20);
		return 0;
	}

	char* filename = new char[strlen(argv[3])];  // Filename arg
	char* media = new char[strlen(argv[2])];
	char* obj = new char[strlen(argv[1])];

	strcpy(filename, argv[3]); // Copy argv into filename
	strcpy(media, argv[2]); // Copy argv into filename
	strcpy(obj, argv[1]); // Copy argv into filename

	if (load_cascades() == -1) return 0;  // Load the cascades

	// Check the object type arg
	if (!strcmp(obj, "-p"))
		is_car = false;
	else if (!strcmp(obj, "-c"))
		is_car = true;
	else {
		printf("ERROR - Invalid object argument"); return 0;
	}

	// Choose image or video process
	if (!strcmp(media,"-v")) {
		if (use_video(filename) == -1) return 0;
	} else if (!strcmp(media, "-i")) {
		if (use_image(filename) == -1) return 0;
		else waitKey(0);
	}
	else {
		printf("ERROR - Invalid media argument"); return 0;
	}

	waitKey(10);

	return 0;
}

int load_cascades()
{
	//-- 1. Load the cascades
	if (!car1_cascade.load(car1_cascade_name)) { printf("--(!)Error loading Car 1 Cascade!\n"); return -1;	}
	if (!car2_cascade.load(car2_cascade_name)) { printf("--(!)Error loading Car 2 Cascade!\n");	return -1;	}
	if (!car2_cascade.load(car3_cascade_name)) { printf("--(!)Error loading Car 3 Cascade!\n");	return -1; }
	if (!ped1_cascade.load(ped1_cascade_name)) { printf("--(!)Error loading Pedestrian 1 Cascade!!\n"); return -1; }
	if (!ped2_cascade.load(ped1_cascade_name)) { printf("--(!)Error loading Pedestrian 2 Cascade!!\n"); return -1; }
	if (!ped3_cascade.load(ped1_cascade_name)) { printf("--(!)Error loading Pedestrian 3 Cascade!!\n"); return -1; }
	return 0;
}

// Use a video file to detect and display
int use_video(char* filename)
{
	VideoCapture capture(filename);
	Mat frame;
	bool paused = false;
	VideoWriter outputVideo;  // Open the output

	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	outputVideo.open(OUTPUT_FILENAME, CV_FOURCC('P', 'I', 'M', '1'), capture.get(CV_CAP_PROP_FPS), S, true);

	if (!outputVideo.isOpened())
		cout << "Could not open output video for write: " << OUTPUT_FILENAME << endl;

	// Try to read the video file
	if (!capture.isOpened()) { cout << "Could not open file " << filename << endl; return -1; }

	while (true) {
		if (!paused) {
			capture >> frame;  // Take the current frame and put it into the Mat object
			++frame_cnt;
			//cout << "Frame: " << frame_cnt << endl;

			if (!frame.empty()) {
				frame = detectAndDisplay(frame);  // Do some detection and tracking
				imshow(window_name, frame); // Output image to display
			}
			else {
				printf("===== No captured frame -- Break! =====\n");
				break;
			}

			outputVideo << frame;  // output the frame to the output file
		}
		int c = waitKey(20);

		if ((char)c == 'c')	break;
		if ((char)c == 'p') paused = true;
	}

	outputVideo.release(); // Release the output

	return 0;
}

// Uses an image to detect an object
int use_image(char* filename)
{
	Mat img = imread(filename, 1);
	if (!img.data) { printf("ERROR - Could not load image \n");	return -1; }

	// Read the image
	if (!img.empty()) {
		img = detectAndDisplay(img);  // Do some detecting
		imshow(window_name, img); // Output image to display
	}
	else {
		printf(" --(!) No Image frame -- Break!");
		return -1;
	}

	//bool imwrite(const String& filename, InputArray img, const vector<int>& params=vector<int>() )
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	imwrite(OUTPUT_IMG_FILENAME, img, compression_params);

	return 0;
}

// Detect and Display Cars
Mat detectAndDisplay(Mat frame)
{
	vector<Rect2d> newObjs;
	std::vector<Rect> someObjs;
	vector<KeyPoint> keypts;
	Mat frame_gray;

	try {

		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		blur(frame_gray, frame_gray, Size(BLUR_SZ, BLUR_SZ), Point(-1, -1));  // Smoothing
		imshow("equalized", frame_gray);

		if (is_car) {
			car1_cascade.detectMultiScale(frame_gray, someObjs, CAR_SCALE_FACTOR, CAR_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, CAR_H_SIZE);
			car2_cascade.detectMultiScale(frame_gray, someObjs, CAR_SCALE_FACTOR, CAR_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, CAR_H_SIZE);
			//car3_cascade.detectMultiScale(frame_gray, someObjs, CAR_SCALE_FACTOR, CAR_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, CAR_H_SIZE);
		}
		else {
			ped1_cascade.detectMultiScale(frame_gray, someObjs, PED_SCALE_FACTOR, PED_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, PED_H_SIZE);
			ped2_cascade.detectMultiScale(frame_gray, someObjs, PED_SCALE_FACTOR, PED_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, PED_H_SIZE);
			ped3_cascade.detectMultiScale(frame_gray, someObjs, PED_SCALE_FACTOR, PED_MIN_NEIGH, CV_HAAR_SCALE_IMAGE, PED_H_SIZE);
		}

		// Check to see if tracked objects list is populated
		if (tracked_objs.size() > 0)
			deleteLostTracks();

		newObjs = compareDetectedToTracked(someObjs, frame);  // Compare the newly detected objects to the existing ones

		if (tracked_objs.size() > 0) {  // draw the tracked objects
			for (unsigned i = 0; i < tracked_objs.size(); i++) {
				Point pt = Point(tracked_objs[i].bnd_box.x, tracked_objs[i].bnd_box.y);
				Size sz(tracked_objs[i].bnd_box.width/2, tracked_objs[i].bnd_box.height/2);
				Point center = Point(tracked_objs[i].centroid.pt.x, tracked_objs[i].centroid.pt.y);
				string text = "id: " + std::to_string(tracked_objs[i].id);
				keypts.push_back(tracked_objs[i].centroid);

				rectangle(frame, tracked_objs[i].bnd_box, Scalar(255, 0, 0), 2, 1);
				//ellipse(frame, center, sz, 90, 0, 360, Scalar(255, 0, 255), 1,8,0);
				cv::putText(frame, text, pt, CV_FONT_NORMAL, 0.5f, Scalar::all(255), 1, 8);
			}
		}

		drawKeypoints(frame, keypts, frame);

		// Go through all of the detected cars with h
		/*for (size_t i = 0; i < someObjs.size(); i++)
		{
			Point topLeft(someObjs[i].x, someObjs[i].y);
			Point bottomRight(someObjs[i].x + someObjs[i].width, someObjs[i].y + someObjs[i].height);
			rectangle(frame, topLeft, bottomRight, Scalar(255, 255, 50), 2, 8, 0);
		}*/

		return frame;
	}
	catch (int e) {
		cout << "Exception: " << e << endl << endl;
		return frame;
	}
}


// Compare the detected objects to the tracked objects
// Takes in a vector of Rects that are the newly detected objects
// Returns nothing, but function updates the tracked objects
vector<Rect2d> compareDetectedToTracked(vector<Rect> objs, Mat frame)
{
	vector<Rect2d> newObjs;
	vector<KeyPoint> objs_kPts;
	vector<int> high_val_indexes;
	vector<float> indexOfOverlapObjValues(objs.size(), 0);  // The overlap value of each obj
	int trk_cnt = tracked_objs.size();
	int det_cnt = objs.size();  // Detection size

	// Get the key points for all of the detected objects
	for (int i = 0; i < objs.size(); ++i) {
		objs_kPts.push_back(rectToKeyPoint(objs[i]));
	}

	//Compute the highest overlap value for each track. Zero means that they don't overlap.
	for (int i = 0; i < trk_cnt; ++i) {
		float highestVal = 0; // Highest index value
		int high_Indx = -1; // Highest index

		for (int j = 0; j < det_cnt; ++j) {
			float val = rectOverLap(tracked_objs[i].bnd_box, objs[j]);  // Get rectangle overlap

			if (highestVal < val) {  // Check if the current value is higher
				highestVal = val;
				high_Indx = j;
			}

			if (indexOfOverlapObjValues[j] < val)
				indexOfOverlapObjValues[j] = val;
		}

		high_val_indexes.push_back(high_Indx);

		// Check the highest value and then update the tracked objects
		if (high_Indx > 0) {
			tracked_objs[i].updateTracker(frame);
			tracked_objs[i].currentlyVisible(); // Update the age and visibility counters
		}
		else {
			tracked_objs[i].currentlyNotVisible();
		}
	}

	// Add new objects to be tracked
	for (int k = 0; k < objs.size(); ++k) {
		if (indexOfOverlapObjValues[k] < OVERLAP_THRESH) {
			Rect2d r2d = rectToRect2d(objs[k]);  // Add a new tracker object
			trackedObj trk(nextId, objs_kPts[k], r2d, TRK_ALG_MF);
			trk.initTracker(frame);

			++nextId;  // Increment id count;
			tracked_objs.push_back(trk);
			newObjs.push_back(r2d);
		}
	}

	return newObjs;
}

// Delete tracked objects that are not to be tracked any more
void deleteLostTracks()
{
	vector<int> lostInds;
	int track_cnt = tracked_objs.size();

	// Compute the fraction of the track's age for which it was visible.
	for (int i = 0; i < track_cnt; ++i) {
		int ageVal = tracked_objs[i].age;
		int lostVal = tracked_objs[i].consecInvCount;
		int totVisCnt = tracked_objs[i].totVisCount;
		float val = (float)totVisCnt / (float)ageVal;

		//cout << "Car id[" << tracked_objs[i].id << "]  Age:[" << ageVal << "]  Invisible:[" << lostVal << "]  Total Vis:[" << totVisCnt << "] Ratio:[" << to_string(val) << " / " << to_string(MAX_VIS_TO_AGE_RATIO) << "]\n";

		if ((ageVal < MAX_AGE && val < MAX_VIS_TO_AGE_RATIO) || (lostVal >= MAX_VIS))
			lostInds.push_back(i); // Add to list of cars to remove
	}

	// Remove the selected cars
	for (int i = 0; i < lostInds.size(); ++i) {
		//cout << "Erasing Index: " << i << " --> Car Id: " << tracked_objs[i].id << endl;
		if (lostInds[i] < tracked_objs.size())
			tracked_objs.erase(tracked_objs.begin() + lostInds[i]);  // Corresponding struct.
	}

	cout << endl;
}

// Convert point to Keypoint
KeyPoint rectToKeyPoint(Rect r)
{
	Point2f pt(r.x+(r.width/2),r.y+(r.height/2));
	return KeyPoint(pt, (r.width / 2), -1, 0, 0, -1);
}

// Convert Rect to Rect2d
Rect2d rectToRect2d(Rect r)
{
	return Rect2d(r.x, r.y, r.width, r.height);;
}

// Calculated the overlap of two rectangles
// Returns the percent value of overlap
float rectOverLap(Rect r1, Rect r2)
{
	Rect r3 = r1 & r2; // Get the intersection
	float r3Area = (float)r3.area();

	float val1 = float(r3Area / r1.area());
	float val2 = float(r3Area / r2.area());

	return ((val1 + val2) / 2);
}