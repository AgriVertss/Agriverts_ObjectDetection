import cv2
print(cv2.__version__)
import numpy as np
import time
from skimage.segmentation import clear_border
from skimage import measure, color, io

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',1320,0)

cv2.createTrackbar('hueLower', 'Trackbars',2,179,nothing)
cv2.createTrackbar('hueUpper', 'Trackbars',75,179,nothing)

# cv2.createTrackbar('hue2Lower', 'Trackbars',179,179,nothing)
# cv2.createTrackbar('hue2Upper', 'Trackbars',179,179,nothing)

cv2.createTrackbar('satLow', 'Trackbars',80,255,nothing)
cv2.createTrackbar('satHigh', 'Trackbars',255,255,nothing)
cv2.createTrackbar('valLow','Trackbars',75,255,nothing)
cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)

def Trackbarpos():
    hueLow=cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp=cv2.getTrackbarPos('hueUpper', 'Trackbars')

    # hue2Low=cv2.getTrackbarPos('hue2Lower', 'Trackbars')
    # hue2Up=cv2.getTrackbarPos('hue2Upper', 'Trackbars')

    Ls=cv2.getTrackbarPos('satLow', 'Trackbars')
    Us=cv2.getTrackbarPos('satHigh', 'Trackbars')

    Lv=cv2.getTrackbarPos('valLow', 'Trackbars')
    Uv=cv2.getTrackbarPos('valHigh', 'Trackbars')

    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])
    return l_b,u_b


def homogeneity(rects):
    areas=np.array([])
    for rect in rects:
        areas=np.append(areas,rect[2]*rect[3])
    homo=np.zeros((len(rects),2))
    mean_area=np.mean(areas, axis=0)
    std_area=np.std(areas, axis=0)
    for idx, area in enumerate(areas):
        per=(mean_area-area)*100/mean_area
        if abs(per) < 66:  homo[idx,:]= per ,0
        elif abs(per) >=66:  homo[idx,:]= per ,1
    return homo

def drwinfo(frame,homo,rects,healty=[]):
    for idx,rect in enumerate( rects):
        boyut=" "
        if len(healty)!=0:
            h=healty[idx]
            text2=f"Bitki saglik degeri %{h:.2f} dir"
            frame = cv2.putText(
              img = frame,
              text = text2,
              org = (rect[0]-10,rect[1]-35),
              fontFace = cv2.FONT_HERSHEY_DUPLEX,
              fontScale = 0.4,
              color = (255, 0, 255),
              thickness = 1
            )
        i=abs(homo[idx,0])

        if homo[idx,0]<0: boyut=" buyuktur"
        elif homo[idx,0]>0: boyut=" kucuktur"
        
        if homo[idx,1]==1: text="!!! Homojenite orantisizligi !!!"
        elif homo[idx,0]!=0: text=f"Bitki ortalamadan %{i:.2f} daha"+ boyut
        else: text="Bitki ortalama boyuttadir"
        frame = cv2.putText(
          img = frame,
          text = text,
          org = (rect[0]-10,rect[1]-10),
          fontFace = cv2.FONT_HERSHEY_DUPLEX,
          fontScale = 0.4,
          color = (255, 0, 255),
          thickness = 1
        )
        
        # cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(255,0,0),3)
    return frame

def homogeneity1(areas):
    areas=np.array(areas)
    homo=np.zeros((len(areas),2))
    mean_area=np.mean(areas, axis=0)
    std_area=np.std(areas, axis=0)
    for idx, area in enumerate(areas):
        per=(mean_area-area)*100/mean_area
        if abs(mean_area-area) < std_area*1.5:  homo[idx,:]= per ,0
        elif abs(mean_area-area) >= std_area*1.5:  homo[idx,:]= per ,1

    return homo


def mask(hsv,l_b,u_b):
    FGmask=cv2.inRange(hsv,l_b,u_b)

    # blur
    FGmask = cv2.GaussianBlur(FGmask,(15,15),0)
            # thresholdq
    thresh = cv2.threshold(FGmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening = clear_border(opening) #Remove edge touching grains

    # dilation
    sure_bg = cv2.dilate(opening,kernel,iterations=10)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    ret2, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255,0)
    
    #Later you realize that 0.25* max value will not separate the cells well.
    #High value like 0.7 will not recognize some cells. 0.5 seems to be a good compromize
    
    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    #Now we create a marker and label the regions inside. 
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0. 
    #For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.
    
    #Now we are ready for watershed filling. 
    markers = cv2.watershed(frame,markers)
    #The boundary region will be marked -1
    
    #Let us color boundaries in yellow. 
    # frame[markers == -1] = [0,255,255]  
    
    img2 = color.label2rgb(markers, bg_label=0)
    
    return img2


                
def green_retio(frame,areas,rects):
    healty=np.zeros(len(rects))
    for idx,rect in enumerate(rects):
        roi=frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        l_b=np.array([40,80,75])
        u_b=np.array([75,255,255])
        mask1=mask(hsv,l_b,u_b)
        contours,hierachy=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        ca=[]
        for c in contours:
            if 100*cv2.contourArea(c)/(rect[2]*rect[3])>2:
                ca.append(cv2.contourArea(c))
        
        if len(ca)>0:
            menaarea=sum(ca)/len(ca)
            healty[idx]=100*menaarea/areas[idx]
    return healty
        
def counters(FGmask,area):
    contours,hierachy=cv2.findContours(FGmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    rect=[]
    areas=[]
    for c in contours:
            hull = cv2.convexHull(c,returnPoints = False)
            defects = cv2.convexityDefects(c,hull)
            # basic contour data
            ca = cv2.contourArea(c)
            # target on contour
            p = 2*100*ca/area
            if (p >= 1) and (p <= 40):
                    M = cv2.moments(c)#;print( M )
                    tx = int(M['m10']/M['m00'])
                    ty = int(M['m01']/M['m00'])
                    cv2.drawContours(frame,[c],0,(100,0,255),2)
                    cv2.circle(frame,(tx,ty), 3,(0,255,255),-1)
                    rect.append(cv2.boundingRect(c))
                    areas.append(cv2.contourArea(c))
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        far = tuple(c[f][0])
                        cv2.line(frame,(tx,ty),far,[255,0,0],1)
                        cv2.circle(frame,far, 3,(0,255,255),-1)


    return areas,rect

Vc="New images/WhatsApp Video 2022-02-28 at 09.30.05.mp4"
cam= cv2.VideoCapture(Vc)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (960, 736))
dispW=640
dispH=480

while cam.isOpened():
    healty=[]
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    width,height,depth = np.shape(frame)
    area = width*height
    hsv=cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2HSV)
    l_b,u_b= Trackbarpos()
    FGmask=mask(hsv,l_b,u_b)
    # areas,rects=counters(FGmask,area=area)
    # if len(areas)>0 and len(rects)>0:
    #     healty=green_retio(frame,areas,rects)
    # homo=homogeneity(rects)
    # if len(healty)==0:
    #     frame=drwinfo(frame, homo, rects)
    # if len(healty)!=0:
    #     frame=drwinfo(frame, homo, rects,healty)
    # FGmask=np.expand_dims(FGmask, axis=2)
    # FGmask=cv2.cvtColor(FGmask,cv2.COLOR_GRAY2BGR)

    result=np.hstack((frame,FGmask))
    # out.write(result) 
    cv2.imshow('nanoCam',result)
    cv2.moveWindow('nanoCam',0,0)
    time.sleep(0.01)
    if cv2.waitKey(1)==ord('q'):
        break
    
cam.release()
out.release() 
cv2.destroyAllWindows()
