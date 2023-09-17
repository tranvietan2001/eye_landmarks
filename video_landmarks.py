# USAGE
# python3 video_landmarks.py --shape_predictor shape_predictor_68_face_landmarks.dat
from imutils import face_utils
import argparse
import cv2 
import dlib
import imutils
from scipy.spatial import distance as dist

threshold = 0.20 # nguong canh bao (co the thay doi tuy vao mat moi nguoi, phu hop nhat laf 0.25): canh bao khi < threshold, ngung canh bao khi >= threshold
couter = 0 
total = 100 # muc canh bao
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B) / (2.0 * C)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")     # dlib’s pre-trained facial landmark detector (phát hiện 68 landmarks)
args = vars(ap.parse_args())

video = cv2.VideoCapture(0)  

# khởi tạo dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()   # dựa trên HOG + Linear SVM tìm face, xem thêm bài face recognition

# Tạo the facial landmerk predictor
predictor = dlib.shape_predictor(args["shape_predictor"])
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
while True:
    ret, frame = video.read()

    # Vẫn phải detect được khuôn mặt trước khi tìm facial landmarks
    # load ảnh, resize, convert to gray (cần cho HOG)
    frame = imutils.resize(frame, width=500)    # giữ nguyên aspect ratio, để size lớn quá lag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # dùng cho HOG detector bên dưới

    # detect faces in the grayscale image
    # nhận 2 tham số ảnh (có thể ảnh màu), 2-nd parameter - số image pyramids tạo ra trước khi detect faces (upsample)
    # nó giúp phóng ảnh lên để có thể phát hiện khuôn mặt nhỏ hơn, dùng thì chạy lâu hơn
    rects = detector(gray, 1)   # trả về list các rectangle chứa khuôn mặt (left, top, right, bottom) <=> (xmin, ymin, xmax, ymax)

    # duyệt qua các detections
    for (i, rect) in enumerate(rects):
        # xác định facial landmarks for the face region sau đó convert các facial landmarks (x,y)
        # về numpy array, mỗi hàng là một cặp tọa độ
        shape = predictor(gray, rect)   # nhận 2 tham số là ảnh đầu vào và vùng phát hiện khuôn mặt, shape.part(i) là cặp tọa độ thứ i

        # chuyển về dạng numpy các coordinates
        shape = face_utils.shape_to_np(shape)   # numpy array (68, 2)

        # Chuyển dlib's rectange (left, top, right, botttom) = (xmin, ymin, xmax, ymax) to OpenCV-style bounding box (xmin, ymin, w, h)
        # Dễ dàng chuyển được 
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)    # vẽ rectangle quanh khuôn mặt

        # hiển thị số khuôn mặt trong ảnh, chú ý ở đây đang duyệt qua các detections
        # cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # duyệt qua các coordinates of facial landmarks (x, y) và vẽ chúng lên ảnh
        for (x, y) in shape:
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR+ rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # cv2.circle(frame, (x, y), 1, (255, 0, 0), -1) //hien 68 diem quanh khung mat
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1) #hien vong quanh mat trai
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)  #hien vong quanh mat phai

            # print(ear)
            if ear >= threshold:
                couter = 0
                print(couter)
                cv2.putText(frame, "Trang thai binh thuong!",(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            else:
                couter+=1
                print(couter)
            if couter > total:
                cv2.putText(frame, "Ngu gat!",(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                couter = 0
            else:
                couter = 0
                cv2.putText(frame, "Trang thai binh thuong!",(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        # hiển thị ảnh đầu ta với face detections + facial landmarks
        cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):     # nhấn "q" để thoát
        break

video.release()
cv2.destroyAllWindows()



