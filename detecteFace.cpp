
#include <stdio.h>
#include <assert.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <objdetect.hpp>

using namespace cv;
using namespace std;

//Variable global
  Mat ci, gsi;
   uchar pixValue;
   Point p1,p2;
 // IplImage* img;
 // IplImage *color;
  CascadeClassifier * face_cc = new CascadeClassifier("haarcascade_frontalface_default.xml");
  CascadeClassifier * eye_cc= new CascadeClassifier("haarcascade_eye.xml");
  CascadeClassifier * mouth_cc= new CascadeClassifier("Mouth.xml");
  CascadeClassifier * nez_cc= new CascadeClassifier("Nariz.xml");

string text = "Visage";
int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
double fontScale = 1;
int thickness = 1;





//fonction detection head
  void visage();
  void yeux();
  void bouche();
  void nez();

int main(){
	 VideoCapture * cap = new VideoCapture(0);

  if(!cap->isOpened()) {
    delete cap;
    cap = new VideoCapture(CV_CAP_ANY);
  }

    

  if((face_cc == NULL && eye_cc==NULL)|| !cap->isOpened()) {
    if(face_cc) delete face_cc;
    if(eye_cc)delete eye_cc;
    if(mouth_cc) delete mouth_cc;
    if(nez_cc)delete nez_cc;
    if(cap) delete cap;
    return 1;
  }

  cap->set(CV_CAP_PROP_FRAME_WIDTH,  420);
  cap->set(CV_CAP_PROP_FRAME_HEIGHT, 340);


  namedWindow("Face detection", CV_WINDOW_NORMAL);
  while(1) {
    *cap >> ci;
 

  cvtColor(ci, gsi, COLOR_BGR2GRAY);

  visage();
	yeux();
	nez();
	bouche();

    imshow("Face detection", ci);
    if((waitKey(10) & 0xFF) == 27)
      break;
  }
  return 0;
}




void visage(){//detecte le visage
	 vector<Rect> faces;
	 face_cc->detectMultiScale(gsi, faces, 1.3, 5);


	 
	  for (vector<Rect>::iterator fc = faces.begin(); fc != faces.end(); ++fc) {
      	rectangle(ci, (*fc).tl(), (*fc).br(), Scalar(0, 0, 255), 2, CV_AA);//rouge
        // centre le text
        Point textOrg(((*fc).tl(), (*fc).br()));
        putText(ci, text, textOrg, fontFace, fontScale,Scalar::all(255), thickness, 8);   
        /* for (int i = 0; i < ci.cols; i++) {
    for (int j = 0; j < ci.rows; j++) {
        Vec3b &intensity = ci.at<Vec3b>(j, i);
        for(int k = 0; k < ci.channels(); k++) {
          pixValue=30;
            intensity.val[k] = pixValue;
        }
     }
}*/
    }
}

void yeux(){
	  vector<Rect>eyes;
	  eye_cc->detectMultiScale(gsi,eyes,1.3,5);

	  for (vector<Rect>::iterator fc = eyes.begin(); fc != eyes.end(); ++fc) {
     	 rectangle(ci, (*fc).tl(), (*fc).br(), Scalar(255, 0, 0), 2, CV_AA);//bleu
        Point textOrg(((*fc).tl(), (*fc).br()));
        putText(ci, "oeil", textOrg, fontFace, fontScale,Scalar::all(255), thickness, 8);
        
         /* for (int i = 0; i < ci.cols; i++) {
    for (int j = 0; j < ci.rows; j++) {
        Vec3b &intensity = ci.at<Vec3b>(j, i);
        for(int k = 0; k < ci.channels(); k++) {
          pixValue=30;
            intensity.val[k] = pixValue;
        }
     }
}*/
    }
}

void bouche(){
	    vector<Rect>mouths;
	       mouth_cc->detectMultiScale(gsi,mouths,1.3,5);

	    for (vector<Rect>::iterator fc = mouths.begin(); fc != mouths.end(); ++fc) {
      rectangle(ci, (*fc).tl(), (*fc).br(), Scalar(255, 0,255), 2, CV_AA);//rose
     Point textOrg(((*fc).tl(), (*fc).br()));
        putText(ci, "bouche", textOrg, fontFace, fontScale,Scalar::all(255), thickness, 8);
         /* for (int i = 0; i < ci.cols; i++) {
    for (int j = 0; j < ci.rows; j++) {
        Vec3b &intensity = ci.at<Vec3b>(j, i);
        for(int k = 0; k < ci.channels(); k++) {
          pixValue=30;
            intensity.val[k] = pixValue;
        }
     }
}*/
    }
}

void nez(){
	vector<Rect>nariz;
	nez_cc->detectMultiScale(gsi, nariz, 1.3, 5);

    for (vector<Rect>::iterator fc = nariz.begin(); fc != nariz.end(); ++fc) {
      rectangle(ci, (*fc).tl(), (*fc).br(), Scalar(0,255, 255), 2, CV_AA);//jaune
       Point textOrg(((*fc).tl(), (*fc).br()));
        putText(ci, "nez", textOrg, fontFace, fontScale,Scalar::all(255), thickness, 8);
     
      //fonction qui change l'intensité des pixels ( pas encore operationelle)
 /* for (int i = 0; i < ci.cols; i++) {
    for (int j = 0; j < ci.rows; j++) {
        Vec3b &intensity = ci.at<Vec3b>(j, i);
        for(int k = 0; k < ci.channels(); k++) {
          pixValue=30;
            intensity.val[k] = pixValue;
        }
     }
}*/
    }
}