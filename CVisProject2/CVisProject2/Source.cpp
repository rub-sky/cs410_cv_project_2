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
#include "opencv2\opencv.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/** Function Headers */
void load_cascades();
void detectAndDisplayCars(Mat frame);
void detectAndDisplayPeds(Mat frame);
void capture_video();
void use_image(Mat img1, Mat img2);

/** Global variables */
String car1_cascade_name = "../haar_cascades/haarcascade_car_1.xml";  // Works for front of cars
String car2_cascade_name = "../haar_cascades/cars3.xml";
//String ped_cascade_name = "../haar_cascades/haarcascade_ped.xml";
String ped_cascade_name = "../haar_cascades/haarcascade_pedestrian.xml";

CascadeClassifier car1_cascade;
CascadeClassifier car2_cascade;
CascadeClassifier ped_cascade;
string window_name = "Car detection";
string window_name_peds = "Pedestrian detection";
RNG rng(12345);

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

void capture_video()
{
	//========
	//CvCapture* capture;
	VideoCapture capture("../video/peds1.mp4");
	Mat frame;

	//-- 2. Read the video stream
	//capture.open(0);
	if (!capture.isOpened())  // check if we succeeded
		return;

	while (true)
	{
		capture >> frame;  // Take the current frame and put it into the Mat object

		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{
			//detectAndDisplayCars(frame);
			detectAndDisplayPeds(frame);
		}
		else {
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		int c = waitKey(2);

		if ((char)c == 'c')
		{
			break;
		}
	}

}

void use_image(Mat img1, Mat img2)
{
	// Read the video stream
	// Apply the classifier to the frame
	if (!img1.empty() && !img2.empty())
	{
		detectAndDisplayCars(img1);
		detectAndDisplayPeds(img2);
	}
	else {
		printf(" --(!) No captured frame -- Break!");
	}

}

// Detect and Display Cars
void detectAndDisplayCars(Mat frame)
{
	std::vector<Rect> cars1;
	std::vector<Rect> cars2;
	std::vector<Rect> peds;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//  Detect cars Front
	car1_cascade.detectMultiScale(frame_gray, cars1, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	car2_cascade.detectMultiScale(frame_gray, cars2, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));


	// Go through all of the detected cars
	for (size_t i = 0; i < cars1.size(); i++)
	{
		Point topLeft(cars1[i].x, cars1[i].y);
		Point bottomRight(cars1[i].x + cars1[i].width, cars1[i].y + cars1[i].height);
		rectangle(frame, topLeft, bottomRight, Scalar(50, 255, 50), 2, 8, 0);
	}

	// Go through all of the detected cars with h
	for (size_t i = 0; i < cars2.size(); i++)
	{
		Point topLeft(cars2[i].x, cars2[i].y);
		Point bottomRight(cars2[i].x + cars2[i].width, cars2[i].y + cars2[i].height);
		rectangle(frame, topLeft, bottomRight, Scalar(255, 255, 50), 2, 8, 0);
	}

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
