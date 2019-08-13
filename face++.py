#################################识别视频或图片中的人脸##############################
import os
os.chdir('D:/DataCenter/AI/TRAIN/NetworkClass/face_detect_database/')
import face_recognition

###############################################################################
#读取文件夹下所有图片名 
known_path = 'D:/DataCenter/AI/TRAIN/NetworkClass/face_detect_database/data'

#读取训练数据
known_face_encodings = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        print(path_name)
        
        if os.path.isdir(full_path):    #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件
            if dir_item.endswith('.jpg') or dir_item.endswith('.png'):
                known_image = face_recognition.load_image_file(full_path)
                try:
                    #读取图片进行编码
                    known_encoding = face_recognition.face_encodings(known_image)[0]
                    known_face_encodings.append(known_encoding)
                    #图片标签
                    labels.append(path_name.split("\\")[-1]) 
                except IndexError:#face_recognition.face_encodings(known_image)编码为空list
                    continue
                    
    return known_face_encodings,labels

#加载已知图片数据
known_face_encodings, labels = read_path(known_path) 

import cv2
import sys
import gc
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
    
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "./haarcascade_frontalface_alt2.xml"    
    
    #循环检测识别人脸
    while True:
        ret, frame = cap.read()   #读取一帧视频
        
        if ret is True:
            
            #图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                
 
        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                
                try:
                    unknown_encoding = face_recognition.face_encodings(image)[0]
                except IndexError:#face_recognition.face_encodings(known_image)编码为空list
                    continue                               
                results = face_recognition.compare_faces(known_face_encodings, unknown_encoding )
                
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                
                for i in range(0, len(results)):
                    if results[i] == True:
                        #文字提示是谁
                        cv2.putText(frame,'%s'%labels[i], 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号
                                    (255,0,255),                           #颜色
                                    2)
                        break #由于循环匹配，匹配成功不跳出循环会导致识别的另一个文字显示在图像上造成文字重叠
                    
                    else:
                        #文字提示是谁
                        cv2.putText(frame,'unknown', 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号q
                                    (255,0,255),                           #颜色
                                    2) 
                        break #同理，匹配不成功不跳出循环会导致识别的另一个文字显示在图像上造成文字重叠
                    
        cv2.imshow("Recognize Me !!!", frame) #不要显示中文，否则窗口标题显示乱码
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(120)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()