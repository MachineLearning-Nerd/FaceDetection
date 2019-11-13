import dlib
import cv2
from drawmarks import renderFace

if __name__ == "__main__":
    # Convert the rectangle in to the tuple of the dlib.rectangle output 
    def point_to_rectangle(rectangle):
        # Take the input from the frontal face detector 
        # e.g facedetector = dlib.get_frontal_face_detector()
        new_rect = dlib.rectangle(int(rectangle.left()),int(rectangle.top()),
                    int(rectangle.right()),int(rectangle.bottom()))
        return new_rect

    # This is the function for the writing the data into the files
    def writelandmarkfile(dlandmarks,landmarks_filesname):
        with open(landmarkfilename,'w') as f :
            for p in dlandmarks.parts():
                f.write("%s %s\n" %(int(p.x),int(p.y)))
        f.close

    # Put the predictor path here which is the pretrained path
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    # Call the face detector
    # This is the face detector. We will first detect the face.
    facedetector = dlib.get_frontal_face_detector()

    # Landmark detector is implemented in the shape predictor class 
    # So we will call it first and then we will go ahead
    landmarkdetector = dlib.shape_predictor(predictor_path)

    # Read the image from thse camera
    cam = cv2.VideoCapture(0)

    # If we just want to input as the image then 
    # imagename = "dinesh.jpg"

    while(True):
        # Capture the video frame by frame
        ret , frame = cam.read()

        # for image as the input  
        # im = cv2.imread(imagename)

        # Landmarks will be stored in the results folder
        landmarkbase = "results/faces"

        # Process of the detection 
        # Detect the face in the image
        facerectangle = facedetector(frame,0)

        # Number of faces detected in the image 
        print("Number of the faces detected:", len(facerectangle))

        # Detect all the landmarks in the image and stores
        landmarkall = []

        if (len(facerectangle)==0):
            # show the image
            cv2.imshow("Facial Landmark detector",frame)
            cv2.waitKey(1)
            continue

        # Loop over the all the face those are detected in the frontal face detector
        for i in range(0,len(facerectangle)):
            # Get the all the point of the rectangle 
            new_rect = point_to_rectangle(facerectangle[i])

            # For every face rectangle run the face landmark detection
            landmarks = landmarkdetector(frame,new_rect)

            # Number of the landmarks that are detected 
            if i==0:
                print("Number of landmarks:",len(landmarks.parts()))
            
            # Stores the all the landmarks 
            landmarkall.append(landmarks)

            # Draw all the land marks 
            renderFace(frame, landmarks)

            landmarkfilename = landmarkbase + "_" + str(i) + ".txt"

            # Write the all the landmarks in the files 
            writelandmarkfile(landmarks,landmarkfilename)

            # show the image
            cv2.imshow("Facial Landmark detector",frame)
            key = cv2.waitKey(1)
        if key == 101 :
            cv2.destroyAllWindows()
            break

    cv2.waitKey(1)
    cv2.destroyAllWindows()