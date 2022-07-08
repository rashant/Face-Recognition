import cv2
import face_recognition as fr
import numpy as np
import os
vid=cv2.VideoCapture(0, cv2.CAP_DSHOW)


known_face_encodings=[]
known_face_names=[]
for i in (os.listdir('D:\Projects\Face Recognition\Faces')):
    face=fr.load_image_file(os.path.join('D:\Projects\Face Recognition\Faces',i))
    known_face_encodings.append(fr.face_encodings(face)[0])
    #ADDING FACE TO DATABASE
    known_face_names.append(i[:str(i).find('.')].replace(" ", "_"))

process_this_frame=True
name="Unknown"
face_locations = []
face_encodings = []
face_names = []
while(True):
    ret,frame=vid.read()
    frame=cv2.flip(frame,1)
    if process_this_frame:
        small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rbg_small_frame=small_frame[:,:,::-1]
        face_locations=fr.face_locations(rbg_small_frame)
        face_encodings=fr.face_encodings(rbg_small_frame,face_locations)

        face_names=[]
        for face_encoding in face_encodings:
            matches=fr.compare_faces(known_face_encodings,face_encoding)
            face_distances=fr.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distances)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame=not process_this_frame
    for(top,right,bottom,left),name in zip(face_locations,face_names):
        top*=4
        right*=4
        bottom*=4
        left*=4

        cv2.rectangle(frame,(left-30,top-60),(right+30,bottom+30),(5,225,0),2)
        cv2.rectangle(frame, (left-30, bottom - 5), (right+30, bottom+30), (5,225,0), cv2.FILLED)
        cv2.putText(frame,name,(left+6,bottom+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1,cv2.FILLED)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
