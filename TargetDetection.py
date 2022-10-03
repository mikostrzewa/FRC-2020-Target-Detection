#Importing libraries
import datetime
import cv2
import numpy as np
import math
import json
from networktables import NetworkTables
import threading
from threading import Thread

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

class WebcamVideoStream:
    def __init__(self, src=0, width=320, height=240):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, width)
        self.stream.set(4, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
    # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()


    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
def zoom(frame,scale):
    if(scale<=1):
        return frame
    height, width, channels = frame.shape

    #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(height/(scale*2)),int(width/(scale*2))

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    frame = cv2.resize(cropped, (width, height)) 
    return frame

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

def nothing(x):
    pass
def send(distance,angle_x,angle_y):
    angle*=angleConst
    print("Distance: {} \n Angle: {}".format(distance,angle))
    sd.putNumber("Ball distance", distance)
    sd.putNumber("Angle_x", angle_x)
    sd.putNumber("Angle_y", angle_y)


cond = threading.Condition()
notified = [False]

with open('TargetParameters.json', 'r') as f:
    data = json.load(f)

print(data)

#Constants

Focus=data['Focus']
angleConst=0.0625
objectWidth=43.18

print("Which mode do you want to use?\n1.Dev \n2.User\n3.Jetson")
while True:
    choice = input("Your choice: ")
    if(choice=='1'):
        mode=1
        break
    elif(choice=='2'):
        mode=2
        break
    elif(choice=='3'):
        mode=3
        break
    else:
        print("Incorrect input!\n\n\n")

focusCalculation = False

answer=input("Do you want to calculate the Focus? (Y/N): ")
if(answer.lower()=='y'):
    dist = int(input("Distance: "))
    Fn=0
    Fsum=0
    focusCalculation=True




if mode==1:
    cv2.namedWindow("Color")
    cv2.createTrackbar("LH", "Color", data['Low_H'], 255, nothing)
    cv2.createTrackbar("LS", "Color", data['Low_S'], 255, nothing)
    cv2.createTrackbar("LV", "Color", data['Low_V'], 255, nothing)
    cv2.createTrackbar("UH", "Color", data['Upper_H'], 255, nothing)
    cv2.createTrackbar("US", "Color", data['Upper_S'], 255, nothing)
    cv2.createTrackbar("UV", "Color", data['Upper_V'], 255, nothing)

    cv2.namedWindow("Noise")
    cv2.createTrackbar("KernelSize", "Noise", data['KernelSize'], 20, nothing)
    cv2.createTrackbar("Iterations", "Noise", data['Iterations'], 20, nothing)
    cv2.createTrackbar("ApproxValue", "Noise", data['ApproxValue'], 100, nothing)


if mode<3:  
    cv2.namedWindow("Zoom")
    cv2.createTrackbar("Scale", "Zoom", 1, 50, nothing)

if mode==3:
    NetworkTables.initialize(server='10.81.27.2')
    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()
        sd = NetworkTables.getTable("Dashboard")
cap = WebcamVideoStream(src=0,width=320,height=240).start()
fps = FPS().start()
while True:
    #Getting values from trackbar

    originalFrame = cap.read()
    frame = originalFrame.copy()

    if mode==1:
        l_h = cv2.getTrackbarPos("LH", "Color")
        l_s = cv2.getTrackbarPos("LS", "Color")
        l_v = cv2.getTrackbarPos("LV", "Color")

        u_h = cv2.getTrackbarPos("UH", "Color")
        u_s = cv2.getTrackbarPos("US", "Color")
        u_v = cv2.getTrackbarPos("UV", "Color")

        KernelSize = cv2.getTrackbarPos("KernelSize", "Noise")
        iterations = cv2.getTrackbarPos("Iterations", "Noise")
        approxValue= cv2.getTrackbarPos("ApproxValue", "Noise")           
    else:
        l_h = data['Low_H']
        l_s = data['Low_S']
        l_v = data['Low_V']

        u_h =data['Upper_H']
        u_s = data['Upper_S']
        u_v = data['Upper_V']

        approxValue=data["ApproxValue"]

        KernelSize = data['KernelSize']
        iterations = data['Iterations']

    if mode<3:
        scale = cv2.getTrackbarPos("Scale", "Zoom")

    if(scale>1):
        frame = zoom(frame,scale)
    #chaniging colors to hsv format
    hsv = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV)

    #Setting hsv range and setting up the mask
    lower_color = np.array([l_h,l_s,l_v])
    upper_color = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #Creating result frame
    result = cv2.bitwise_and(originalFrame, originalFrame, mask=mask)

    #Morphological operations
    kernel = np.ones((KernelSize,KernelSize),np.uint8)
    result = cv2.erode(result,kernel,iterations = iterations)
    result = cv2.dilate(result,kernel,iterations = iterations)
    
    #Result to grey
    imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    #Detecting contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        #Approxying contours
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(result, [approx], 0, (0, 0, 255), 3)
        #Checking if circle

        if (len(approx)in range(6,9)):
            maxX=-1
            minX=100000
            minY=100000
            maxY=-1
            for coordinates in approx:
                maxX=max(coordinates[0][0],maxX)
                minY=min(coordinates[0][1],minY)
                minX=min(coordinates[0][0],minX)
                maxY=max(coordinates[0][1],maxY)
            #Calculating distance
            cX=int((maxX+minX)/2)
            cY=minY
            r=maxY-minY
            distance=(objectWidth*Focus)/(r)

            if(focusCalculation):
                Fsum+=(dist*r)/objectWidth
                Fn+=1

            distance=int(distance)
            angle_x=frame.shape[1]/2-cX
            angle_y=frame.shape[0]/2-cY
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 3)
            cv2.circle(frame, (cX, cY), 3, (165,37, 140), -1)
            cv2.putText(frame,"{} cm".format(distance), (cX, cY+15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165,37, 140), 2)
            cv2.putText(frame,"{} px X axis".format(angle_x), (cX, cY+30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165,37, 140), 2)
            cv2.putText(frame,"{} px Y axis".format(angle_y), (cX, cY+45),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165,37, 140), 2)
            if mode == 3:
                send(distance,angle_x,angle_y)

    #cv2.imshow("gray", thresh)
    if mode==1 or mode==2:
        cv2.imshow("frame", frame)
        if mode == 1:
            cv2.imshow("res", result)
        #Key to exit
        key = cv2.waitKey(1)
        if key == 27:
            break
        fps.update()

#Clean up
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.stop()
cv2.destroyAllWindows()
if(focusCalculation):
    Focus=Fsum/Fn
print("Focus: {}".format(Focus))

if mode==1 or focusCalculation:
    outputNoise={"Low_H" : l_h,"Low_S" : l_s,"Low_V" : l_v,"Upper_H" : u_h,"Upper_S" : u_s,"Upper_V" : u_v,"KernelSize": KernelSize, "Iterations" : iterations, "ApproxValue": approxValue, "Focus":Focus}
    print(outputNoise)

    decision = input("Do you want to save the changes?(Y/N): ")
    if(decision=="Y" or decision=="y"):
        noiseSettings = json.dumps(outputNoise)
        f = open("TargetParameters.json","w")
        f.write(noiseSettings)
        f.close()
        print("File saved successfully!")
