import numpy as np
import cv2
import imutils as iu

# powierzchnia sudoku na obrazie wejsciowym musi miec minimum 40000px
# klawisz ESC zamyka okna

# True - image z kemry
# False - image z pliku test/test.jpg, jest kilka iinych w katalogu
cam = False


"""Pobiera ostatnia klatkę ze streamu z kamery, ESC wyzwala akcję"""
if cam == True:
    capture = cv2.VideoCapture(1)  #czasami zmiana na wartość -1, 0, 1, włącza kamerę
    ret, frame = capture.read()

    while(True):
        ret, frame = capture.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 27:  #ESC zamyka okna
            break

    img = cv2.imwrite("frame.jpg",frame)
    cv2.destroyAllWindows()
    image = cv2.imread('frame.jpg')
else:
    image = cv2.imread('test/test.jpg')
    
   
def unsharp_mask(image, kernel_size=(5, 5), sigma=0.5, amount=1, threshold=2):
    """Zwraca zaostrzoną wersję obrazu, używając maski wyostrzającej."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def proc_unsharp_mask():
    """Zwraca zaostrzoną wersję obrazu, używając maski wyostrzającej."""
    sharpened_image = unsharp_mask(image)
    image_gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('frame_unsharp_mask.jpg', image_gray)


#wczytuje frame_unsharp_mask.jpg przechwycony obraz i aplikuje kilka metod, tutaj jest czysta zabawa
def proc_bitwise_not():
    """Kilka metod do przetestowania"""
    image = cv2.imread('frame_unsharp_mask.jpg')
    proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #proc = cv2.medianBlur(proc,5)
    proc = cv2.GaussianBlur(proc, (5, 5), 3)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    proc = cv2.bitwise_not(proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    proc = cv2.dilate(proc, kernel)
    cv2.imwrite("frame_bitwise_not.jpg", proc)

def rotateCorners(corners):
    tl = None # top left
    bl = None # bottom left
    tr = None # top right
    br = None # bottom right

    biggest = 0
    smallest = 1000000
    rest = []

    for corner in corners:
        added = corner[0][0] + corner[0][1]
        if added > biggest:
            biggest = added
            br = corner[0]
        if added < smallest:
            smallest = added
            tl = corner[0]

    for corner in corners:
        if not np.array_equal(corner[0], br) and not np.array_equal(corner[0], tl):
            rest.append(corner[0])
    if len(rest) == 2:
        if rest[0][0] > rest[1][0]:
            bl = rest[1]
            tr = rest[0]
        else:
            bl = rest[0]
            tr = rest[1]

    print ("top-left: %a"%tl)
    print ("bottom-left: %a"%bl)
    print ("top-right: %a"%tr)
    print ("bottom-right: %a"%br)

    return [[tl], [bl], [tr], [br]]


def captureAndCrop():
    img = cv2.imread('frame_bitwise_not.jpg') 
    height, width = img.shape[:2]
    if height > 800 or width > 800:
        if height > width:
            captured = iu.resize(img, height=800)
        else:
            captured = iu.resize(img, width=800)
    else:
        captured = img

    gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
  
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,5)
   
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    biggest = None
    for i in contours:
        area = cv2.contourArea(i)
        if area > 40000:
            epsilon = 0.1*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            #cv2.drawContours(self.captured, [i], 0, (0,0,255), 1)
            if area > maxArea and len(approx)==4:
                maxArea = area
                biggest = i
                corners = approx
            

    if biggest is not None:        
        pts1 = np.float32(rotateCorners(corners))
        pts2 = np.float32([[0,0],[0,450],[450,0],[450,450]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        cuttedThresh = cv2.warpPerspective(thresh,M,(450,450))
        cuttedOrig = cv2.warpPerspective(captured,M,(450,450))

        #cv2.drawContours(captured, [biggest], 0, (0,255,0), 3)
        #cv2.imwrite('/contour.png', captured)
        cv2.imwrite('frame_cuttedThresh.png', cuttedThresh)

    
def show_images():
    while(True):
        frame_unsharp_mask = cv2.imread('frame_unsharp_mask.jpg')
        frame_bitwise_not = cv2.imread('frame_bitwise_not.jpg')
        frame_cuttedThresh = cv2.imread('frame_cuttedThresh.png')
        
        cv2.imshow("Original", image)
        cv2.imshow("Frame_unsharp_mask", frame_unsharp_mask)
        cv2.imshow("Frame_bitwise_not", frame_bitwise_not)
        cv2.imshow("Frame_cuttedThresh", frame_cuttedThresh)

        if cv2.waitKey(1) == 27:  #ESC zamyka okna
            break

    cv2.destroyAllWindows()


proc_unsharp_mask()
proc_bitwise_not()
captureAndCrop()
show_images()