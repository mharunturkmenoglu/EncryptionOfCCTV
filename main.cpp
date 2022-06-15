/// \brief  main function. Neural Network whith OpenCV and DNN and GPU are deployed. Tracks the object frame by frame based on the CocoDataSet.
/// \attention remove comments in lines 44 and 45 to run the program on GPU
/// \param  argc An integer argument count of the command line arguments
/// \param  argv An argument vector of the command line arguments
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include "encryption.h"
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;
using namespace dnn;
int main(int, char**) {
    string file_path = "Models/";
    vector<string> class_names;
    ifstream ifs(string(file_path + "object_detection_classes_coco.txt").c_str());
    string line;
    // Load in all the classes from the file
    while (getline(ifs, line))
    {
        //cout << line << endl;
        class_names.push_back(line);
    }
    // Read in the neural network from the files
    auto net = readNet(file_path + "ssd_mobilenet_v2_coco_2018_03_29.pb",
        file_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", "TensorFlow");
    // Open up the webcam
    VideoCapture cap("Footage.mp4");
    // Run on either CPU or GPU
    //net.setPreferableBackend(DNN_BACKEND_CUDA);
    //net.setPreferableTarget(DNN_TARGET_CUDA);
    // Set a min confidence score for the detections
    float min_confidence_score = 0.5;
    // Loop running as long as webcam is open and "q" is not pressed
    while (cap.isOpened()) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // Load in an image
        Mat image;
        bool isSuccess = cap.read(image);


        //hist(image);


        if (!isSuccess) {
            cout << "Could not load the image!" << endl;
            break;
        }
        int image_height = image.cols;
        int image_width = image.rows;
        auto start = getTickCount();
        // Create a blob from the image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
            true, false);
        // Set the blob to be input to the neural network
        net.setInput(blob);
        // Forward pass of the blob through the neural network to get the predictions
        Mat output = net.forward();
        auto end = getTickCount();
        // Matrix with all the detections
        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        // Run through all the predictions
        for (int i = 0; i < results.rows; i++) {
            int class_id = int(results.at<float>(i, 1));
            float confidence = results.at<float>(i, 2);
            // Check if the detection is over the min threshold and then draw bbox
            if (confidence > min_confidence_score) {
                int bboxX = int(results.at<float>(i, 3) * image.cols);
                int bboxY = int(results.at<float>(i, 4) * image.rows);
                int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
                int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);

                Mat feedback = image;
                if (bboxX + bboxWidth < image_width && bboxY + bboxHeight < image_height) {
                    //feedback = encrypt(feedback, bboxY, bboxX, bboxHeight, bboxWidth);
                    feedback = encrypt2(feedback, bboxY, bboxX, bboxHeight, bboxWidth);
                    //feedback = stepByStep(feedback, bboxY, bboxX, bboxHeight, bboxWidth);
                }
                //image = encrypt3(image);



                /*Mat img1 = image.clone();
                int m = img1.rows;
                int n = img1.cols;
                int ch = img1.channels();
                NPCR_UACI(img1, image, m, n, ch);*/

                
                //hist(image);
                
                
                
            }
        }
        auto totalTime = (end - start) / getTickFrequency();
        putText(image, "FPS: " + to_string(int(1 / totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
        imshow("image", image);
        int k = waitKey(1);
        if (k == 113) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
}
