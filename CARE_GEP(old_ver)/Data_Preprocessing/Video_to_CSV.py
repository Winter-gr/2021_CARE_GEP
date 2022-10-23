"""
    Class VideoToCSV
    
    Developer: Joo Hyeong Lee
    Date: 2021.10.14
    ver 1.2
    
    update 1.0 : 2021.10.04
    update 1.1 : 2021.10.08
    update 1.2 : 2021.10.14
        - Add collaboration with 'dir popup class'
    update 1.3 : 2021.10.15
        - Video cropping, saturation and value revise
        - csv saving code added
        
        
    input:
        - Exact dir to 'video' files
    output:
        - CSV with keypoint landmarks(each rows = keypoint, each col = x, y, z)
    
    Point:
        - To clearly load bbox from csvs(which were already made), 
            dir should be correct below(check def conversion)
"""

'''
    함수 및 클래스 고려사항
        - 스켈레톤 인식 안 되는 것은 어떻게 처리할 것인지 -> 일단 이미지 아웃 필요할듯?
        - 모든 프레임을 다 처리할 것인지(일단 10fps로 하자.)
    
    작성 중 나중에 추가해야할 것 같은 사항
        - folder 리스트 불러올 때 정렬 기준
        - float 자리수를 줄이면 연산이 빨라지려나?
        - open을 미리 해서... csv 파일 0B 짜리를 다 만들고 시작해버린다. 문제인가?     
'''

### Import
import Dir_popup_class as popup

import os
import numpy as np
import pandas as pd
import json
import csv
import time
import cv2

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose



class InValidException(Exception):
    def __init__(self):
        super(InValidException, self).__init__("Class Initialize Failed")

        
        
class ConvertToCSV(object):
    """From video files, extract skeleton data and convert into csv files"""
    
    def __init__(self, out_folder, video_format, verbose = False):
        print("ConvertToCSV initialized")
        
        # Variables setting
        self._csv_dir = out_folder
        self._video_format = video_format
        self._verbose = verbose
        self._init_flag = False
        self._exit_flag = 0
        # Exception
        if self._video_format != 'jpg' and self._video_format != 'mp4':
            print("ERROR: file format should be jpg or mp4")
            raise InValidException
        
        
        # Print for checking
        #if self._verbose:
            #print('video list check: ', self._video_lists)
            #print('action name: ', self._action_name)

            

    def Extract_Skeleton(self, file_dir):
        self._file_dir = file_dir
        
        if self._video_format == 'jpg':
            self.Conversion_jpg()
        else:
            self.Conversion_mp4()
        
        return self._exit_flag   
    
    def Conversion_mp4(self):      
        '''
        mp4_fold_dir = "../../AI_hub_second/video_action_3"
        mp4_example_dir = "../../AI_hub_second/video_action_3/3-1"
        csv_out_dir = "../../AI_hub_second/Json_to_csv_module_TEST/video_action_3"
        
        file dir = "../../AI_hub_second/video_action_3/3-1/3-1_001-C01.mp4"
        '''
        
        _bright_value = 40 # 밝기
        _saturation_value = 68 # 채도
        _v_value = 3 # 명도
        _sub_blue = -5 # 파란색 빼기
        
        
        """ 1. Directory(for saving .csv) setting """
        if self._init_flag == False:
            self._init_flag = True
            _tmp_csv_dir = 'video_action_3'
            save_csv_dir = os.path.join(self._csv_dir, _tmp_csv_dir)
            # Ex. save_csv_dir = "../../.AI_hub_second/Json_to_csv_module_TEST/video_action_3"
        
        # Extract 2-1... 2-10... for mkdir
        _tmp_dir_iter_len = len('../../AI_hub_second/video_action_3')
        _mp4_name = self._file_dir[_tmp_dir_iter_len +1:] # 2-10/2-10_C01_xxxx.mp4
        
        _last_slash = _mp4_name.find('/')
        _tmp_dir_iter = _mp4_name[:_last_slash]# == 2-1 ... 2-10...

        _csv_folder = os.path.join(save_csv_dir, _tmp_dir_iter)
        # Ex. '../../AI_hub_second/Json_to_csv_module_TEST/video_action_3/3-12'
        
        if not os.path.exists(_csv_folder):
            os.makedirs(_csv_folder)
        
        # For making new csv file
        open_csv_dir = os.path.join(save_csv_dir, _mp4_name)
        # Ex. '../../AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1/2-12/2-12_C01_xxxx.json

        # Join the path: for csv files
        csv_out_dir = open_csv_dir[:-4] + ".csv"
        csv_out = open(csv_out_dir, 'w', newline='')
        csv_out_writer = csv.writer(csv_out, delimiter = ',', quoting=csv.QUOTE_MINIMAL)

        
        """ 2. Bbox Loading """
        _tmp_bbox_csv_dir = "../../AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1"
        # _mp4_name = self._file_dir[_tmp_dir_iter_len +1:] # 2-10/2-10_C01_xxxx.mp4
        bbox_csv_dir = os.path.join(_tmp_bbox_csv_dir, _mp4_name)
        # bbox_csv_dir = "../../.AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1/3-1/3-1_001-C01.mp4"
        bbox_csv_dir = bbox_csv_dir[:-4] + "_2D_bbox.csv"
        
        bbox_read = pd.read_csv(bbox_csv_dir, header = None, delimiter = ",") 
        
        
        
        """ 3. Play the video """
        cap = cv2.VideoCapture(self._file_dir)
        
        print("mp4 dir: ", self._file_dir)
        # Variable setting
        frame_count = 0
        row_each = 0
        _first_five = True
        _first_nine = True
        
        while cap.isOpened():
            # print('cap is opened')
            ret, img = cap.read()
            skeleton_coord = []
            frame_count += 1

            if ret == False:
                print('video ends')
                break;
            
            # Video is 30fps
            elif frame_count % 10 == 4:
                # print("frame count: ", frame_count)
                if _first_five == True:
                    # first 5th frame has no bbox info
                    _first_five = False
                else:
                    """ 3-2. Video Crop """
                    skeleton_coord = [frame_count]
                    image = img[frame_ly:frame_ry, frame_lx:frame_rx]
                   
                    
                    """ 3-3. Video Color, Saturation, Value revise """
                    _bright_array = np.full(image.shape, (_bright_value - _sub_blue,_bright_value,_bright_value), dtype=np.uint8)
                    _saturation_array = np.full(image.shape, (0,_saturation_value,0), dtype =np.uint8)
                    _value_array = np.full(image.shape, (0,0,_v_value), dtype =np.uint8)
                    
                    img_revised = cv2.add(image, _bright_array)

                    img_revised = cv2.cvtColor(img_revised, cv2.COLOR_BGR2HSV)
                    img_revised = cv2.add(img_revised, _saturation_array)
                    img_revised = cv2.add(img_revised, _value_array)

                    img_revised = cv2.cvtColor(img_revised, cv2.COLOR_HSV2BGR)

                    
                    """ 3-4. Mediapipe Applying """
                
                    img_revised = cv2.cvtColor(img_revised, cv2.COLOR_BGR2RGB)

                    with mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.8) as pose_tracker:
                        result = pose_tracker.process(image = img_revised)
                        pose_landmarks = result.pose_landmarks

                    #if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=img_revised,
                        landmark_list=pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)

                    img_revised = cv2.cvtColor(img_revised, cv2.COLOR_RGB2BGR)
                    
                    """ 4. Save into CSV file """
                    # Save pose landmark info
                    if pose_landmarks is not None:
                        skeleton_tmp = str(pose_landmarks).split('\n')
                        for c in skeleton_tmp:

                            if c == '}' or c == 'landmark {' or c == '' or c.startswith('v', c.find('v')):
                                continue

                            else:
                                coord = float(c[5:])
                                coord = round(coord, 6)

                                # Append each landmark x y z
                                skeleton_coord.append(coord)
                        
                        # Write csv each row for every pose_landmarks(rounded)
                        csv_out_writer.writerow(skeleton_coord)
                                
                    # Showing the image
                    cv2.imshow(_mp4_name, img_revised)
                    
                    _keyboard = cv2.waitKey(10) & 0xFF
                    if (_keyboard == ord('q')) or (_keyboard == ord('a')):
                        if (_keyboard == ord('a')):
                            self._exit_flag = -1
                        break
                    #elif cv2.waitKey(10) & 0xFF == ord('a'):

                    #    break
                
            # Bbox information is valid
            elif frame_count % 10 == 9:
                skeleton_coord = [frame_count]
                
                """ 3-1. Bbox loads """
                if _first_nine == True:
                    _first_nine = False
                    # 2019 bbox = 왼쪽 상단 xy, 오른쪽 상단 xy
                    bbox_frame_number = int(bbox_read.iloc[row_each][0])
                    frame_lx, frame_ly, frame_rx, frame_ry = bbox_read.iloc[row_each][1:5] # 인덱스 1부터 4까지 리턴.
                    frame_lx, frame_ly, frame_rx, frame_ry = self.Frame_resize(frame_lx, frame_ly, frame_rx, frame_ry)
                else:
                    _tmp_frame_lx, _tmp_frame_ly, _tmp_frame_rx, _tmp_frame_ry = bbox_read.iloc[row_each][1:5]
                    _tmp_frame_lx, _tmp_frame_ly, _tmp_frame_rx, _tmp_frame_ry = self.Frame_resize(_tmp_frame_lx, _tmp_frame_ly, _tmp_frame_rx, _tmp_frame_ry)
                
                    if _tmp_frame_lx < frame_lx:
                        frame_lx = _tmp_frame_lx
                    if _tmp_frame_rx > frame_rx:
                        frame_rx = _tmp_frame_rx
                    if _tmp_frame_ly < frame_ly:
                        frame_ly = _tmp_frame_ly
                    if _tmp_frame_ry > frame_ry:
                        frame_ry = _tmp_frame_ry
                        
                        
                """ 3-2. Video Crop """
                
                image = img[frame_ly:frame_ry, frame_lx:frame_rx]
                
                ''''
                어떤 bbox는 0프레임부터 시작하는 경우가 존재하여 일단 ' 처리하였음
                # Exception
                if bbox_frame_number != frame_count:
                    print("frame_count: ",frame_count, "bbox count: ", bbox_frame_number)
                    print("ERROR: bbox frame number is not equal to frame count!!")
                    raise InValidException
                '''
                                
                """ 3-3. Video Color, Saturation, Value revise """
                
                _bright_array = np.full(image.shape, (_bright_value - _sub_blue,_bright_value,_bright_value), dtype=np.uint8)
                _saturation_array = np.full(image.shape, (0,_saturation_value,0), dtype =np.uint8)
                _value_array = np.full(image.shape, (0,0,_v_value), dtype =np.uint8)

                img_revised = cv2.add(image, _bright_array)

                img_revised = cv2.cvtColor(img_revised, cv2.COLOR_BGR2HSV)
                img_revised = cv2.add(img_revised, _saturation_array)
                img_revised = cv2.add(img_revised, _value_array)

                img_revised = cv2.cvtColor(img_revised, cv2.COLOR_HSV2BGR)
                
                
                """ 3-4. Mediapipe Applying """
                
                img_revised = cv2.cvtColor(img_revised, cv2.COLOR_BGR2RGB)
                        
                with mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.8) as pose_tracker:
                    result = pose_tracker.process(image = img_revised)
                    pose_landmarks = result.pose_landmarks

                #if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=img_revised,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
                
                img_revised = cv2.cvtColor(img_revised, cv2.COLOR_RGB2BGR)
                
                """ 4. Save into CSV file """
                # Save pose landmark info
                if pose_landmarks is not None:
                    skeleton_tmp = str(pose_landmarks).split('\n')
                    for c in skeleton_tmp:

                        if c == '}' or c == 'landmark {' or c == '' or c.startswith('v', c.find('v')):
                            continue

                        else:
                            coord = float(c[5:])
                            coord = round(coord, 6)

                            # Append each landmark x y z
                            skeleton_coord.append(coord)
                    
                    # Write csv each row for every pose_landmarks(rounded)
                    csv_out_writer.writerow(skeleton_coord)
               
            
                # Showing the image
                cv2.imshow(_mp4_name, img_revised)
                
                
                # Row value increase
                row_each = row_each + 1
                
                
                _keyboard = cv2.waitKey(10) & 0xFF
                if (_keyboard == ord('q')) or (_keyboard == ord('a')):
                    if (_keyboard == ord('a')):
                        self._exit_flag = -1
                    break
                #elif cv2.waitKey(10) & 0xFF == ord('a'):
                    
                #    break
                    
            else:
                if self._verbose:
                    print('video ', cv2_path, ' ended')
        
        cap.release()
        cv2.destroyAllWindows() 
        return self._exit_flag
        
    def Reverse_Extract_Skeleton(self, path):
        '''
            return:
                reveresed frames of video of path in arg
        '''
        _path = path
        
        cap = cv2.VideoCapture(path)
        if cap.isOpened() == False:
            print('none video')
            
        ret, frame = cap.read()

        frame_num = 0
        ret = True
        reversed_frame = []

        while(ret):
            if frame_num%3 == 0:
                frame_resize = cv2.resize(frame, (720,480), interpolation = cv2.INTER_AREA)
                #cv2.imwrite("frame%d.jpg" %frameNum, frameResize)
                
                reversed_frame.append(frame_resize)
                frame_num += 1

                # For checking Resize and Frame downgrade
                #cv2.imshow('resize',frameResize)
                #cv2.waitKey(50)

            else:
                frame_num += 1

            ret, frame = cap.read()

        # 이 방식으로 reversed_frame 에 저장 시, 리스트에 마지막까지 저장된다.
        # lastFrame = reversed_frame.pop()
        print("reversed_frame size", len(reversed_frame))

        reversed_frame.reverse()
 
        # mediapipe pose model
        
        
        for frame_rev in reversed_frame:

            # Image preprocessing for mediapipe
            img = cv2.cvtColor(frame_rev, cv2.COLOR_BGR2RGB)
   
            with mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.7) as pose_tracker:
                result = pose_tracker.process(image = img)
                pose_landmarks = result.pose_landmarks
             
            # If pose is detected
            if result.pose_landmarks is not None:
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # print("detected")
            else:
                print("no pose detected")

            # Showing
            cv2.imshow("reversed frame", img)
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            

            cv2.waitKey(30)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


            '''
                1초에 30frame 인 비디오에서, 1/3프레임만 사용한다면,
                1frame 이 1/30초를 담당하고 있었으므로, 영상 길이를 똑같이 만들고 싶다면
                1frame 이 1/10초를 담당해야 한다.
            '''

            '''    
            while cap.isOpened():
                ret, frame = cap.read()
                cv2.imshow('frame', frame)

                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
            '''

        cap.release()
        cv2.destroyAllWindows()
        
        
    def Frame_resize(self, lx, ly, rx, ry):
        _lx = lx
        _ly = ly
        _rx = rx
        _ry = ry
        if _lx > 0 and _lx < 20:
            _lx = 0
        elif _lx > 40:
            _lx = _lx - 30
            
        if _ly > 0 and _ly < 20:
            _ly = 0
        elif _ly > 40:
            _ly = 40
            
        '''elif _ly > 40 and _ly < 60:
            _ly = 40
        elif _ly > 60:
            _ly = _ly - 50'''
            
        if _rx > 1900 and _rx < 1920:
            _rx = 1920
        elif _rx < 1870:
            _rx = _rx + 40
            
        if _ry < 1080 and _ry > 1060:
            _ry = 1080
        elif _ry < 1050:
            _ry = 1050
        
        '''elif _ry < 1040 and _ry > 1020:
            _ry = 1040
        elif _ry < 1020:
            _ry = _ry + 50'''
        
        if _lx < 0:
            _lx = 0
        if _ly < 0:
            _ly = 0
        if _rx < 0:
            _rx = 0
        if _ry < 0:
            _ry = 0
            
        return _lx, _ly, _rx, _ry

