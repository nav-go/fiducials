
/*
 * Copyright (c) 2017, Ubiquity Robotics
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project.
 *
 */

#include "fiducial_slam/map.h"
#include "fiducial_slam/estimate.h"

/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void getSingleMarkerObjectPoints(float markerLength, vector<Point3f>& objPoints) {

    CV_Assert(markerLength > 0);

    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.push_back(Vec3f(-markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f,-markerLength / 2.f, 0));
    objPoints.push_back(Vec3f(-markerLength / 2.f,-markerLength / 2.f, 0));
}

// Euclidean distance between two points
static double dist(const cv::Point2f &p1, const cv::Point2f &p2)
{
    double x1 = p1.x;
    double y1 = p1.y;
    double x2 = p2.x;
    double y2 = p2.y;

    double dx = x1 - x2;
    double dy = y1 - y2;

    return sqrt(dx*dx + dy*dy);
}

// Compute area in image of a fiducial, using Heron's formula
// to find the area of two triangles
static double calcFiducialArea(const std::vector<cv::Point2f> &pts)
{
    const Point2f &p0 = pts.at(0);
    const Point2f &p1 = pts.at(1);
    const Point2f &p2 = pts.at(2);
    const Point2f &p3 = pts.at(3);

    double a1 = dist(p0, p1);
    double b1 = dist(p0, p3);
    double c1 = dist(p1, p3);

    double a2 = dist(p1, p2);
    double b2 = dist(p2, p3);
    double c2 = c1;

    double s1 = (a1 + b1 + c1) / 2.0;
    double s2 = (a2 + b2 + c2) / 2.0;

    a1 = sqrt(s1*(s1-a1)*(s1-b1)*(s1-c1));
    a2 = sqrt(s2*(s2-a2)*(s2-b2)*(s2-c2));
    return a1+a2;
}

// estimate reprojection error
double Estimation::getReprojectionError(const vector<Point3f> &objectPoints,
                            const vector<Point2f> &imagePoints,
                            const Vec3d &rvec, const Vec3d &tvec) {

    vector<Point2f> projectedPoints;

    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix,
                      distortionCoeffs, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    double rerror = totalError/objectPoints.size();
    return rerror;
}


void Estimation::estimatePoseSingleMarker(const vector<Point2f> &corners, 
       Vec3d &rvec, Vec3d &tvec, double &reprojectionError)
{
    vector<Point3f> markerObjPoints;
    getSingleMarkerObjectPoints(fiducialLen, markerObjPoints);

    cv::solvePnP(markerObjPoints, corners, cameraMatrix, distortionCoeffs, rvec, tvec);

    reprojectionError =
          getReprojectionError(markerObjPoints, corners, rvec, tvec);
}


Estimation::Estimation(double fiducialLen)
{
    haveCaminfo = false;

    this->fiducialLen = fiducialLen;

    // Camera intrinsics
    cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

    // distortion coefficients
    distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);
}


void Estimation::camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    if (haveCaminfo) {
        return;
    }

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            cameraMatrix.at<double>(i, j) = msg->K[i*3+j];
        }
    }

    for (int i=0; i<5; i++) {
        distortionCoeffs.at<double>(0,i) = msg->D[i];
    }

    haveCaminfo = true;
    frameId = msg->header.frame_id;
}


void Estimation::estimatePose(const fiducial_msgs::FiducialArray::ConstPtr& msg, 
                              vector<Observation> &observations, 
                              fiducial_msgs::FiducialTransformArray &outMsg)
{
    if (!haveCaminfo) {
        if (frameNum > 5) {
            ROS_ERROR("No camera intrinsics");
        }
        return;
    }

    for (int i=0; i<msg->fiducials.size(); i++) {

        const fiducial_msgs::Fiducial& fid = msg->fiducials[i];

        vector<Point2f > corners;
        corners.push_back(Point2f(fid.x0, fid.y0));
        corners.push_back(Point2f(fid.x1, fid.y1));
        corners.push_back(Point2f(fid.x2, fid.y2));
        corners.push_back(Point2f(fid.x3, fid.y3));

        Vec3d rvec, tvec;
        double reprojectionError;

        estimatePoseSingleMarker(corners, rvec, tvec, reprojectionError);

/*
        aruco::drawAxis(cv_ptr->image, cameraMatrix, distortionCoeffs,
                        rvecs, tvecs, fiducialLen);
*/

        ROS_INFO("Detected id %d T %.2f %.2f %.2f R %.2f %.2f %.2f", fid.fiducial_id,
                 tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2]);

        double angle = norm(rvec);
        Vec3d axis = rvec / angle;
        ROS_INFO("angle %f axis %f %f %f", angle, axis[0], axis[1], axis[2]);

        tf2::Quaternion q;
        q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);

        // Convert image_error (in pixels) to object_error (in meters)
        double objectError =
            (reprojectionError / dist(corners[0], corners[2])) *
            (norm(tvec) / fiducialLen);

        tf2::Transform T(q, tf2::Vector3(tvec[0], tvec[1], tvec[2]));

        Observation obs(fid.fiducial_id,
                        tf2::Stamped<TransformWithVariance>(TransformWithVariance(
                                T, objectError), msg->header.stamp, frameId),
                        reprojectionError,
                        objectError);

        observations.push_back(obs);

        fiducial_msgs::FiducialTransform ft;
        ft.fiducial_id = fid.fiducial_id;

        ft.transform.translation.x = tvec[0];
        ft.transform.translation.y = tvec[1];
        ft.transform.translation.z = tvec[2];

        ft.transform.rotation.w = q.w();
        ft.transform.rotation.x = q.x();
        ft.transform.rotation.y = q.y();
        ft.transform.rotation.z = q.z();

        ft.fiducial_area = calcFiducialArea(corners);
        ft.image_error = reprojectionError;
        ft.object_error = objectError;

        outMsg.transforms.push_back(ft);
    }
}
