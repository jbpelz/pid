# USAGE
# python create_pupil_samples_n.py --face cascades/haarcascade_frontalface_default.xml --video eye.mov

# import the necessary packages
from __future__ import print_function
import numpy as np
# import contourUtils as cntUt
# from pyimagesearch.facedetector import FaceDetector
import imutils
import argparse
import cv2
import time
from matplotlib import pyplot as plt


blurDiameter = 5
CannyThresh1 = 30
CannyThresh2 = 150

PCHARM = True  ## or False, depending on whether it will be command line or through PyCharm

VERBOSE = False  # print to stdout
PLOT = True  # use to inhibit plots to display

BIN_SIZE = 16


RED     = (0, 0, 255)
GREEN   = (0,255,0)
BLUE    = (255,0,0)
YELLOW  = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN    = (255, 255,0)
WHITE   = (255,255,255)
BLACK   = (0,0,0)

COLORS = [GREEN, RED, BLUE, YELLOW, MAGENTA, CYAN, WHITE]

font1 = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_DUPLEX


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face", required = True,
#     help = "path to where the face cascade resides")
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the face detector
# fd = FaceDetector(args["face"])

# if a video path was not supplied, grab the reference
# to the gray

if not PCHARM:
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
else:
    camera = camera = cv2.VideoCapture('eye.mov')

print("camera = {}".format(camera))
print("camera.get(cv2.CAP_PROP_FRAME_WIDTH) = {}".format(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
if camera.get(cv2.CAP_PROP_FRAME_WIDTH) > 640:
    frameTooLarge = True
else:
    frameTooLarge = False


cv2.waitKey(1000)

# keep looping
frameNumber = 0

sleepTime = 0.002  # Start with a sleepTime of 1 msec

if PLOT:
    pltFig = plt.figure(1)
    plt.ion()  # interactive on - nonblocking plots

pupilOrCRlikelihood = 0

if VERBOSE: print("****************************************************************************************")
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    if frameTooLarge:
        frame = imutils.resize(frame, width = 320)

    time.sleep(sleepTime)

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame and convert it to grayscale
    # frame = imutils.resize(frame, width = 640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edge = cv2.Canny(blurred, 50, 85)

    (_, contours, _) = cv2.findContours(edge.copy(),
         cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # retrieve ALL contours
#        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if VERBOSE: print("Frame number {} has {} contours".format(frameNumber, len(contours)))

    colorI = 0
    for contourNum in range(1, len(contours)):
        cont = contours[contourNum]

        moments = cv2.moments(cont)
        centroidX = int(moments['m10']/(moments['m00']+1e-99))
        centroidY = int(moments['m01']/(moments['m00']+1e-99))

        contourPerimeter = cv2.arcLength(cont, False)
        contourArea = cv2.contourArea(cont)

        if contourPerimeter != 0:
            contourAreaToPerimeter = contourArea/contourPerimeter
        else:
            contourAreaToPerimeter = 0.0

        if contourAreaToPerimeter > 0.75:  # Maybe a pupil?
            # print("cnt: [{:3d}] (len{:3d}): ctr: ({:3d},{:3d}), perim: {:4.1f}, area: {:6.1f}, A/P = {:3.1f})".
            # format(contourNum, len(cont),
            # centroidX, centroidY, contourPerimeter,
            # contourArea, contourAreaToPerimeter))

            txt = "contour[%3d] (length %3d): centroid = (%3d,%3d), perimeter = %5.1f, area = %5.1f, A/P = %4.1f)" \
                  % (contourNum, len(cont),
            centroidX, centroidY, contourPerimeter, contourArea,
            contourAreaToPerimeter)

            cv2.drawContours(frame, cont, -1, COLORS[colorI], -1)
            lineX=centroidX
            lineY=centroidY

            x,y,w,h = cv2.boundingRect(cont)

            if len(cont) > 4:  # Probably a pupil?
                pupilOrCRlikelihood = 1
                ellipse = cv2.fitEllipse(cont)
                (ellipseX, ellipseY),(majorAxis, minorAxis), ellipseAngle = ellipse


                ellipseTxt = 'FR#%5d: ellipse ctr = (%5.1f, %5.1f) axes = (%4.1f, %4.1f)  angle = %4.1f' \
                             % (frameNumber, ellipseX, ellipseY, majorAxis, minorAxis, ellipseAngle)

                # To get the mean value of the 'inside' & 'outside':
                insideEllipseMask  = np.zeros(gray.shape,np.uint8)  # black copy of image
                outsideEllipseMask = np.zeros(gray.shape,np.uint8)  # black copy of image
                cv2.drawContours(insideEllipseMask,[cont],0,255,-1)

                cv2.drawContours(insideEllipseMask,[cont],0,255,-1)  # Draw the contour inside the region

                cv2.ellipse(outsideEllipseMask, ellipse, 255, 11)  # Draw an ellipse outside the region
                cv2.drawContours(outsideEllipseMask,[cont],0,0,-1)  # Blank the contour inside the region

                if VERBOSE: cv2.imshow("Inside & Outside EllipseMasks", np.hstack([insideEllipseMask, outsideEllipseMask]))
                # cv2.waitKey(0)

                # pixelpoints = np.transpose(np.nonzero(InsideEllipseMask))

                # print("pixelpoints = {}".format(pixelpoints))

                insideMeanStDevVal = cv2.meanStdDev(gray, mask = insideEllipseMask)
                outsideMeanStDevVal = cv2.meanStdDev(gray, mask = outsideEllipseMask)

                insideMeanVal = insideMeanStDevVal[0][0][0]
                insideStdDevVal = insideMeanStDevVal[1][0][0]

                outsideMeanVal = outsideMeanStDevVal[0][0][0]
                outsideStdDevVal = outsideMeanStDevVal[1][0][0]

                if VERBOSE: print('-----------------------------------------------------------------------------')
                if VERBOSE: print("cnt: [{:3d}] (len{:3d}): ctr: ({:3d},{:3d}), perim: {:4.1f}, area: {:6.1f}, A/P = {:3.1f})".
                    format(contourNum, len(cont),
                    centroidX, centroidY, contourPerimeter,
                    contourArea, contourAreaToPerimeter))

                if VERBOSE: print("  =======>> Ellipse: ctr: ({:5.1f},{:5.1f}) MA,ma: ({:4.1f}, {:4.1f})  angle: {:5.1f} \
   mean, stdDev: Inside/Outside/Ratio: ({:5.1f}, {:5.1f}) / ({:5.1f}, {:5.1f}) / ({:3.2f}, {:3.2f})".
                    format(ellipseX, ellipseY, majorAxis, minorAxis, ellipseAngle,
                           insideMeanVal, insideStdDevVal, outsideMeanVal, outsideStdDevVal,
                           insideMeanVal/outsideMeanVal, insideStdDevVal/outsideStdDevVal))

            else:
                pupilOrCRlikelihood = 0

            colorI = colorI + 1

            if colorI > 6:
                colorI = 0

            convexHull = cv2.convexHull(cont)  # calculate convex Hull
            cv2.drawContours(frame, convexHull, -1, MAGENTA, -1)

    # show our detected contours
    dispImage = np.hstack([frame, cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)])

    if pupilOrCRlikelihood:
        insideOutsideRatio = insideMeanVal/outsideMeanVal

        if insideOutsideRatio < 0.55:  # Probably a pupil
            color = BLACK
            # Calculate the histogram of the pupil area
            if PLOT:
                pupilHist = cv2.calcHist([gray], [0], insideEllipseMask, [256/BIN_SIZE], [0,256])
                xVal=np.arange((BIN_SIZE//2), 256-(BIN_SIZE//2)+1, BIN_SIZE)
                print(xVal)

                plt.cla()
                print("len(xVal) = {}   len(pupilHist) = {}".format(len(xVal), len(pupilHist)))

                plt.plot(xVal,pupilHist,'k', linewidth=3)
                plt.xlim([0,256])
                plt.xticks(np.arange(0,256+1,BIN_SIZE))
                plt.show()

        elif insideOutsideRatio > 1.85:  # Probably a CR
            color = WHITE


        else:
            color = YELLOW

        cv2.line(dispImage, (lineX, 0), (lineX,240), color)
        cv2.line(dispImage, (0, lineY), (320, lineY), color)

        cv2.rectangle(dispImage,(x, y), (x+w, y+h), (0, 0, 128),1)
        cv2.ellipse(dispImage,ellipse,(0,128,0),1)

    dispImage = imutils.resize(dispImage, width = 1280)

    # Write some Text
    cv2.putText(dispImage, txt ,(10,450), font1, 0.8, (0,150,0) ,2)
    cv2.putText(dispImage, ellipseTxt ,(10,425), font1, 0.8, (0, 0, 180), 2)

    cv2.imshow('Edges and contours', dispImage)

    # if the 'q' key is pressed, stop the loop
    retVal = cv2.waitKey(1)

    if VERBOSE: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   {}".format(retVal))

    if retVal & 0xFF == ord("q"):
        print("I caught a Q")
        break

    elif retVal & 0xFF == ord("s"):
        sleepTime = min(0.5, sleepTime*2)
        print("slower; sleep {:.0f} msec".format(sleepTime*1000))

    elif retVal & 0xFF == ord("f"):
        sleepTime = max(0.001, sleepTime/2)
        print("faster; sleep {:.0f} msec".format(sleepTime*1000))

    elif retVal & 0xFF == ord(" "):
        print("wait for another keypress ...")
        cv2.waitKey(0)

    time.sleep(sleepTime)

    frameNumber = frameNumber + 1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

