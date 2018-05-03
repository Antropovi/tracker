#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

#include "eco.h"

using namespace std;
using namespace cv;

bool gotBB = false;
bool drawing_box = false;
cv::Rect2d box;
void mouseHandler(int event, int x, int y, int flags, void *param)
{
    switch (event) {
    case EVENT_MOUSEMOVE:
        if (drawing_box) {
            box.width = x - box.x;
            box.height = y - box.y;
        }
        break;
    case EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cv::Rect(x, y, 0, 0);
        break;
    case EVENT_LBUTTONUP:
        drawing_box = false;
        if (box.width < 0) {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0) {
            box.y += box.height;
            box.height *= -1;
        }
        gotBB = true;
        break;
    }
}

void drawBox(cv::Mat &image, cv::Rect box, cv::Scalar color, int thick)
{
    rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}

int main(int argc, char *argv[])
{


    VideoCapture video("/home/salen/Videos/TestVid.avi");
    // Exit if video is not opened
    if (!video.isOpened()) {
        cout << "Could not read video file" << endl;
        return 1;

    }
    namedWindow("Test", WINDOW_AUTOSIZE);
    setMouseCallback("Test", mouseHandler, NULL);

    cv::Mat frame;
    video >> frame;

    cv::Mat temp;
    frame.copyTo(temp);
    while (!gotBB) {
        drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
        cv::imshow("Test", frame);
        temp.copyTo(frame);
        if (waitKey(20) == 27)
            return 1;
    }
    //************** Remove callback  *********************************
    setMouseCallback("Test", NULL, NULL);

    int64 tick1 = cv::getTickCount();
    ECO tr(1);
    tr.init(frame, box);

    int64 tick2 = cv::getTickCount();
    int64 res = tick2 - tick1;
    int num = 0;

    auto writer = cv::VideoWriter("MyEcoCppRes1234456.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame.size());

    while (num++ < 10000) {
        video >> frame;

        tick1 = cv::getTickCount();
        tr.update(frame, box);

        tick2 = cv::getTickCount();
        res += tick2 - tick1;

        cv::Mat resframe = frame.clone();
        cv::rectangle(resframe, box, cv::Scalar(0, 255, 0));

        putText(resframe, "FPS   " + to_string((double)num / ((double)res / cv::getTickFrequency())), Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        cv::imshow("Test", resframe);
        cv::waitKey(1);
        writer << resframe;
        cout << "Number is " << num << endl;
    }

    cout << "Hello World!" << endl;
    return 0;
}


////#include <opencv2/opencv.hpp>
////#include <opencv2/tracking.hpp>
////#include <opencv2/core/ocl.hpp>
////#include <deque>
////#include "eco.h"

////using namespace cv;
////using namespace std;

////struct sTracker{
////    Ptr<Tracker> tracker;
////    Rect2d bbox;
////    bool ok;
////};

////struct sTracker1{
////    Ptr<ECO> tracker;
////    Rect2d bbox;
////    bool ok;
////};


////int main(int argc, char **argv)
////{
////    // List of tracker types in OpenCV 3.2
////    // NOTE : GOTURN implementation is buggy and does not work.
////    string trackerTypes[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "ECO"};
////    // vector <string> trackerTypes(types, std::end(types));

////    // Create a tracker
////    string trackerType = trackerTypes[6];

////    Ptr<Tracker> tracker;

////    deque<sTracker1> allTrackers;

////    // Read video
////    VideoCapture video("/home/salen/Videos/original.avi");

////    // Exit if video is not opened
////    if(!video.isOpened())
////    {
////        cout << "Could not read video file" << endl;
////        return 1;

////    }

////    ifstream input("/home/salen/Videos/detected_vehicles_output.txt");
////    string line;
////    Rect2d tempFromFile;
////    sTracker1 temp;



////    // Read first frame
////    Mat frame;
////    bool ok = video.read(frame);
////    auto writer = cv::VideoWriter("Test240418.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame.size());
////    int numOfFrame = 0;

////    int name= 0;

////    bool flag = false;

////    while(video.read(frame))
////    {
////            getline(input, line);
////            getline(input, line);

////            while(line != ""){
////              istringstream cub(line);
////              cub >> tempFromFile.x >> tempFromFile.y >> tempFromFile.width >> tempFromFile.height;

////              for (sTracker1 &tr : allTrackers){
////                  if ((tr.bbox & tempFromFile).area() > 0 && tr.ok){
////                      flag = true;
////                  }
////              }

////              if (!flag && numOfFrame > 600) {
////                  temp.bbox = tempFromFile;

//////                  if (trackerType == "BOOSTING")
//////                      temp.tracker = TrackerBoosting::create();
//////                  if (trackerType == "MIL")
//////                      temp.tracker = TrackerMIL::create();
//////                  if (trackerType == "KCF")
//////                      temp.tracker = TrackerKCF::create();
//////                  if (trackerType == "TLD")
//////                      temp.tracker = TrackerTLD::create();
//////                  if (trackerType == "MEDIANFLOW")
//////                      temp.tracker = TrackerMedianFlow::create();
////                  if (trackerType == "GOTURN")
////                      temp.tracker = TrackerGOTURN::create();


////                  //temp.tracker = TrackerKCF::create();

////                  ++name;
////                  temp.tracker = new ECO(name);
////                  (*temp.tracker).init(frame, tempFromFile);
////                  temp.ok = true;
////                  allTrackers.push_back(temp);
////              }

////              flag = false;

////              getline(input, line);

////        }

////        // Update the tracking result

////        auto tr = allTrackers.begin();
////        for(;tr != allTrackers.end();){
////          (*tr).ok = (*tr).tracker->update(frame, (*tr).bbox);

////          if ((*tr).ok) {
////              rectangle(frame, (*tr).bbox, Scalar( 255, 0, 0 ), 2, 1 );
////              tr++;
////          } else {
////              tr = allTrackers.erase(tr);
////          }
////        }

////        // Display tracker type on frame
////        putText(frame, trackerType + " Tracker NumOfFrame " + to_string(numOfFrame), Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

////        if ( numOfFrame > 600) writer << frame;

////        // Display frame.
////        imshow("Tracking", frame);

////        numOfFrame++;

////        // Exit if ESC pressed.
////        int k = waitKey(30);
////        if(k == 27)
////        {
////            break;
////        }

////    }
////}

