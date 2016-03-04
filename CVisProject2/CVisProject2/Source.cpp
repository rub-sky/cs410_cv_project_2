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

const int MAX_VIS = 20;
const int MAX_AGE= 6;
const int MIN_VIS = 8;
const double MAX_VIS_TO_AGE_RATIO = 0.6;
const string TRK_ALG_KCF = "KCF";
const string TRK_ALG_MF = "MEDIANFLOW";
const string TRK_ALG_TLD = "TLD";

struct trackedObj {
	int id = 0;
	KeyPoint centroid;
	Rect2d bnd_box;
	int age = 1;
	int totVisCount = 1;
	int consecInvCount = 0;
	Ptr<Tracker> trkr;

	// Default constructor
	trackedObj(int idNum, KeyPoint cent, Rect2d bBox, int ageNum, int tvc, int cic, string alg)
	{
		id = idNum;
		centroid = cent;
		bnd_box = bBox;
		age = ageNum;
		totVisCount = tvc;
		consecInvCount = cic;

		trkr = Tracker::create(alg); // Initializes the tracking algorithm for the object
	}

	// Secondary constructor
	trackedObj(int idNum, KeyPoint cent, Rect2d bBox, string alg)
	{
		id = idNum;
		centroid = cent;
		bnd_box = bBox;

		trkr = Tracker::create(alg);  // Initializes the tracking algorithm for the object
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
};

/** Function Headers */
void load_cascades();
void detectAndDisplay(Mat frame);
void detectAndDisplayPeds(Mat frame);
void capture_video();
void use_image(Mat img1, Mat img2);
// For Tracked objecs
vector<Rect2d> compareDetectedToTracked(vector<Rect> objs);
KeyPoint rectToKeyPoint(Rect r);
Rect2d rectToRect2d(Rect r);
void deleteLostTracks();
void set_params();


/** Global variables */
String car1_cascade_name = "../haar_cascades/haarcascade_car_1.xml";  // Works for front of cars
String car2_cascade_name = "../haar_cascades/cars3.xml";
//String car2_cascade_name = "../haar_cascades/haarcascade_ped.xml";
String ped_cascade_name = "../haar_cascades/haarcascade_pedestrian.xml";
//String car2_cascade_name = "../haar_cascades/haarcascade_fullbody.xml";
String videoName = "../video/video2.avi";  //"../video/mitsubishi_768x576.avi";//"../video/CarTraffic1.mp4";

CascadeClassifier car1_cascade;
CascadeClassifier car2_cascade;
CascadeClassifier ped_cascade;
string window_name = "Car detection";
string window_name_peds = "Pedestrian detection";
RNG rng(12345);
int frame_cnt = 0;

/* For tracker*/
std::string trackingAlg = "MEDIANFLOW"; // KCF, MEDIANFLOW or TLD
MultiTracker trackers(trackingAlg);
vector<Rect2d> objects;
vector<trackedObj> trackers_objs;
int nextId = 0;

// Blob Detector
// Setup SimpleBlobDetector parameters.
SimpleBlobDetector::Params params;
SimpleBlobDetector blobDetect;


/** @function main */
int main(int argc, const char** argv)
{
	// \image_data\static_images
	Mat imgCar = imread("../image_data/static_images/sedan1.jpg", 1);
	//Mat imgCar = imread("../image_data/static_images/traffic.jpg", 1);
	//Mat imgCar = imread("../image_data/static_images/SUV_peds.jpg", 1);

	Mat imgPed = imread("../image_data/static_images/0236.jpg", 1);

	if (!imgCar.data || !imgPed.data) { printf("Error loading src1 \n"); return -1; }

	load_cascades();
	set_params();

	capture_video();
	//use_image(imgCar); // Use an image instead of a video
	//use_image(imgCar, imgCar);

	waitKey(0);
	return 0;
}

void load_cascades()
{
	//-- 1. Load the cascades
	if (!car1_cascade.load(car1_cascade_name))
	{
		printf("--(!)Error loading  Car 1 Cascade!\n");
		return;
	}

	if (!car2_cascade.load(car2_cascade_name))
	{
		printf("--(!)Error loading Car 2 Cascade!\n");
		return;
	}

	if (!ped_cascade.load(ped_cascade_name))
	{
		printf("--(!)Error loading Pedestrian Cascade!!\n");
		return;
	}
}

void set_params()
{
	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 1500;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
}

void capture_video()
{
	//========
	//CvCapture* capture;
	VideoCapture capture(videoName);
	Mat frame;
	bool paused = false;

	VideoWriter outputVideo;  // Open the output
	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

	outputVideo.open("./result.avi", CV_FOURCC('P', 'I', 'M', '1'), capture.get(CV_CAP_PROP_FPS), S, true);

	if (!outputVideo.isOpened()) {
		cout << "Could not open the output video for write: " << videoName << endl;
	}

	//-- 2. Read the video stream
	//capture.open(0);
	if (!capture.isOpened())  // check if we succeeded
		return;

	while (true)
	{
		if (!paused) {
			capture >> frame;  // Take the current frame and put it into the Mat object
			++frame_cnt;
			//cout << "Frame: " << frame_cnt << endl;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty()) {
				detectAndDisplay(frame);
				//detectAndDisplayPeds(frame);
			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			outputVideo << frame;  // output the frame to the output file
		}
		int c = waitKey(10);

		if ((char)c == 'c')	break;
		if ((char)c == 'p') paused = true;
	}
	outputVideo.release(); // Release the output
}

void use_image(Mat img1, Mat img2)
{
	// Read the video stream
	// Apply the classifier to the frame
	if (!img1.empty() && !img2.empty())
	{
		detectAndDisplay(img1);
		detectAndDisplayPeds(img2);
	}
	else {
		printf(" --(!) No captured frame -- Break!");
	}

}

// Detect and Display Cars
void detectAndDisplay(Mat frame)
{
	vector<Rect2d> newObjs;
	std::vector<Rect> someObjs;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	car2_cascade.detectMultiScale(frame_gray, someObjs, 1.25, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

	// ======== start tracking
	deleteLostTracks();
	newObjs = compareDetectedToTracked(someObjs);

	//update the tracking result
	if (newObjs.size() > 0) {
		trackers.add(frame, newObjs);
	}

	trackers.update(frame);

	// draw the tracked object
	for (unsigned i = 0; i < trackers.objects.size(); i++) {
		rectangle(frame, trackers.objects[i], Scalar(255, 0, 0), 2, 1);
		Point pt(trackers.objects[i].x, trackers.objects[i].y);
		string text = "id: " + std::to_string(i);
		cv::putText(frame, text, pt, CV_FONT_NORMAL, 0.5f, Scalar::all(255), 1, 8);
	}
	// ======== end tracking

	// Go through all of the detected cars with h
	/*for (size_t i = 0; i < cars2.size(); i++)
	{
		Point topLeft(cars2[i].x, cars2[i].y);
		Point bottomRight(cars2[i].x + cars2[i].width, cars2[i].y + cars2[i].height);
		rectangle(frame, topLeft, bottomRight, Scalar(255, 255, 50), 2, 8, 0);
	}*/

	imshow(window_name, frame); // Output image to display
}

// Detect and Display Pedestrians
void detectAndDisplayPeds(Mat frame)
{
	std::vector<Rect> peds;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	ped_cascade.detectMultiScale(frame_gray, peds, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < peds.size(); i++)
	{
		Point topLeft(peds[i].x, peds[i].y);
		Point bottomRight(peds[i].x + peds[i].width, peds[i].y + peds[i].height);

		rectangle(frame, topLeft, bottomRight, Scalar(0, 0, 255), 2, 8, 0);
	}
	
	imshow(window_name_peds, frame); // Output the image to display
}

// Compare the detected objects to the tracked objects
// Takes in a vector of Rects that are the newly detected objects
// Returns nothing, but function updates the tracked objects
vector<Rect2d> compareDetectedToTracked(vector<Rect> objs)
{
	vector<Rect2d> newObjs;
	vector<KeyPoint> objs_kPts;
	vector<float> indexOfOverlapObjValues(objs.size(), 0);  // The overlap value of each obj
	int trk_cnt = trackers.objects.size();
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
			float val = KeyPoint::overlap(trackers_objs[i].centroid, objs_kPts[j]); // Get 0 if they dont overlap

			if (highestVal < val) {  // Check if the current value is higher
				highestVal = val;
				high_Indx = j;
			}

			if (indexOfOverlapObjValues[j] < val)
				indexOfOverlapObjValues[j] = val;
		}

		// Check the highest value and then update the tracked objects
		if (high_Indx > 0) {
			trackers_objs[i].centroid = objs_kPts[high_Indx]; // Update the struct tracker obj
			trackers_objs[i].updateBBox();  // Update the bounding box for the tracked object
			trackers_objs[i].currentlyVisible(); // Update the age and visibility counters

			trackers.objects[i] = rectToRect2d(objs[high_Indx]);  // update the tracker object
		}
		else {
			trackers_objs[i].currentlyNotVisible();
		}
	}

	// Add new objects to be tracked
	for (int k = 0; k < objs.size(); ++k) {
		if (indexOfOverlapObjValues[k] == 0) {
			// Add a new tracker object
			Rect2d r2d = rectToRect2d(objs[k]);

			trackedObj trkO(nextId, objs_kPts[k], r2d, TRK_ALG_MF);
			++nextId;  // Increment id count;

			trackers_objs.push_back(trkO);
			newObjs.push_back(r2d);
		}
	}

	return newObjs;
}

// Delete tracked objects that are not to be tracked any more
void deleteLostTracks()
{
	if (trackers.objects.empty())
	{
		trackers_objs.clear();
		return;
	}

	vector<int> lostInds;
	int track_cnt = trackers.objects.size();

	// Compute the fraction of the track's age for which it was visible.
	for (int i = 0; i < track_cnt; ++i) {
		int ageVal = trackers_objs[i].age;
		int lostVal = trackers_objs[i].consecInvCount;
		int totVisCnt = trackers_objs[i].totVisCount;
		float val = (float)totVisCnt / (float)ageVal;

		cout << "Car id[" << trackers_objs[i].id << "]  Age:[" << ageVal << "]  Invisible:[" << lostVal << "]  Total Vis:[" << totVisCnt << "] Ratio:[" << to_string(val) << " / " << to_string(MAX_VIS_TO_AGE_RATIO) << "]\n";

		//if ((ageVal < MAX_AGE && val < MAX_VIS_TO_AGE_RATIO) || (lostVal >= MAX_VIS))
			//lostInds.push_back(i); // Add to list of cars to remove
	}

	// Remove the selected cars
	for (int i = 0; i < lostInds.size(); ++i) {
		cout << "Erasing Index: " << i << " --> Car Id: " << trackers_objs[i].id << endl;
		trackers_objs.erase(trackers_objs.begin() + lostInds[i]);  // Corresponding struct.
		trackers.objects.erase(trackers.objects.begin() + lostInds[i]); // Delete tracked objects
	}

	cout << endl;
}

// Convert point to Keypoint
KeyPoint rectToKeyPoint(Rect r)
{
	Point2f pt(r.x+(r.width),r.y+(r.height));
	return KeyPoint(pt, (r.width / 2), 0, 0, 0, 0);
}

// Convert Rect to Rect2d
Rect2d rectToRect2d(Rect r)
{
	return Rect2d(r.x, r.y, r.width, r.height);;
}