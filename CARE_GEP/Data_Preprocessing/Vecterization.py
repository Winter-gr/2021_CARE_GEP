"""
    ver 1.1
        - divided into ML ver, prediction ver.
    ver 1.2
        - inheritance func added.

"""


import pandas as pd
import numpy as np
import csv
from datetime import datetime


class Normalize_Landmarks(object):
    
    def __init__(self):
        self._keypoint_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]
    

    def Get_Distance_By_Name(self, landmarks, _from, _to):
        landmark_from = landmarks[self._keypoint_names.index(_from)]
        landmark_to = landmarks[self._keypoint_names.index(_to)]
        _dist =  landmark_to - landmark_from
        #_dist = _dist.tolist()
        return _dist

    def Get_Average_By_Name(self, landmarks, _from, _to):
        landmark_from = landmarks[self._keypoint_names.index(_from)]
        landmark_to = landmarks[self._keypoint_names.index(_to)]
        _avg =  (landmark_to + landmark_from) *0.5
        #_avg = _avg.tolist()
        return _avg


    def Distance_Info(self, landmarks): # Feature 24개

        _keypoints_dist = [
            # self.Get_Average_By_Name(landmarks, 'left_hip', 'right_hip') - 
            # self.Get_Average_By_Name(landmarks, 'left_shoulder', 'right_shoulder'),

            self.Get_Distance_By_Name(landmarks, 'left_shoulder', 'left_elbow'),
            self.Get_Distance_By_Name(landmarks, 'right_shoulder', 'right_elbow'),

            self.Get_Distance_By_Name(landmarks, 'left_elbow', 'left_wrist'),
            self.Get_Distance_By_Name(landmarks, 'right_elbow', 'right_wrist'),

            self.Get_Distance_By_Name(landmarks, 'left_hip', 'left_knee'),
            self.Get_Distance_By_Name(landmarks, 'right_hip', 'right_knee'),

            self.Get_Distance_By_Name(landmarks, 'left_knee', 'left_ankle'),
            self.Get_Distance_By_Name(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self.Get_Distance_By_Name(landmarks, 'left_shoulder', 'left_wrist'),
            self.Get_Distance_By_Name(landmarks, 'right_shoulder', 'right_wrist'),

            self.Get_Distance_By_Name(landmarks, 'left_hip', 'left_ankle'),
            self.Get_Distance_By_Name(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self.Get_Distance_By_Name(landmarks, 'left_hip', 'left_wrist'),
            self.Get_Distance_By_Name(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self.Get_Distance_By_Name(landmarks, 'left_shoulder', 'left_ankle'),
            self.Get_Distance_By_Name(landmarks, 'right_shoulder', 'right_ankle'),

            self.Get_Distance_By_Name(landmarks, 'left_hip', 'left_wrist'),
            self.Get_Distance_By_Name(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self.Get_Distance_By_Name(landmarks, 'left_elbow', 'right_elbow'),
            self.Get_Distance_By_Name(landmarks, 'left_knee', 'right_knee'),

            self.Get_Distance_By_Name(landmarks, 'left_wrist', 'right_wrist'),
            self.Get_Distance_By_Name(landmarks, 'left_ankle', 'right_ankle'),
        ]
        # print(len(_keypoints_dist))

        return _keypoints_dist


    def Get_Angle_By_Name(self, landmarks, _nameA, _nameB, _nameC, _nameD):
        vector_1 = self.Get_Distance_By_Name(landmarks, _nameA, _nameB)
        vector_2 = self.Get_Distance_By_Name(landmarks, _nameC, _nameD)
        #print(vector_1, vector_2)

        _product = np.dot(vector_1, vector_2)
        _len_A = np.linalg.norm(vector_1)
        _len_B = np.linalg.norm(vector_2)

        _cos_theta = _product/(_len_A * _len_B)

        _angle = np.arccos(_cos_theta)
        _angle = np.degrees(_angle)
        
        _angle = _angle.tolist()
        # print(_angle)
        return _angle


    def Angle_Info(self, landmarks): # Feature = 28개
        # input = (-1, 3) shape of list
        _keypoints_angle = [
            ##### upper body
            #### One joint
            # left
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "left_shoulder", "left_shoulder", "left_elbow"),
            self.Get_Angle_By_Name(landmarks, "left_hip", "left_shoulder", "left_shoulder", "left_elbow"),
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "left_elbow", "left_elbow", "left_wrist"),

            # right
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "right_shoulder", "right_shoulder", "right_elbow"),
            self.Get_Angle_By_Name(landmarks, "right_hip", "right_shoulder", "right_shoulder", "right_elbow"),
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "right_elbow", "right_elbow", "right_wrist"),


            #### Two joint
            # left
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"),
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "left_hip", "left_elbow", "left_wrist"),
            self.Get_Angle_By_Name(landmarks, "right_hip", "left_hip", "left_shoulder", "left_elbow"),

            # right
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "right_shoulder", "right_elbow", "right_wrist"),
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "right_hip", "right_elbow", "right_wrist"),
            self.Get_Angle_By_Name(landmarks, "left_hip", "right_hip", "right_shoulder", "right_elbow"),


            #### Three joint
            # left
            self.Get_Angle_By_Name(landmarks, "right_hip", "left_hip", "left_elbow", "left_wrist"),

            # right
            self.Get_Angle_By_Name(landmarks, "left_hip", "right_hip", "right_elbow", "right_wrist"),


            ##### lower body
            #### One joint
            # left
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "left_hip", "left_hip", "left_knee"),
            self.Get_Angle_By_Name(landmarks, "right_hip", "left_hip", "left_hip", "left_knee"),
            self.Get_Angle_By_Name(landmarks, "left_hip", "left_knee", "left_knee", "left_ankle"),

            # right
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "right_hip", "right_hip", "right_knee"),
            self.Get_Angle_By_Name(landmarks, "left_hip", "right_hip", "right_hip", "right_knee"),
            self.Get_Angle_By_Name(landmarks, "right_hip", "right_knee", "right_knee", "right_ankle"),


            #### Two joint
            # left
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "left_shoulder", "left_hip", "left_knee"),
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "left_hip", "left_knee", "left_ankle"),
            self.Get_Angle_By_Name(landmarks, "right_hip", "left_hip", "left_knee", "left_ankle"),

            # right
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "right_shoulder", "right_hip", "right_knee"),
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "right_hip", "right_knee", "right_ankle"),
            self.Get_Angle_By_Name(landmarks, "left_hip", "right_hip", "right_knee", "right_ankle"),


            #### Three joint
            # left
            self.Get_Angle_By_Name(landmarks, "right_shoulder", "left_shoulder", "left_knee", "left_ankle"),

            # right
            self.Get_Angle_By_Name(landmarks, "left_shoulder", "right_shoulder", "right_knee", "right_ankle"),

        ]
        return _keypoints_angle


    def Normalize(self, _csv_out_dir, _dir = None, _pose = None, _iter_num = None):
        '''
            _dir = str
            
            _pose = (-1, 3) shape list
            _iter_num = how many times this func is called (valid only for _pose is not None)
            
            if _dir is not none: READ CSV
            else if _pose is not none: READ REAL-TIME POSE LANDMARKS
            
            What saved as csv file will be sent to Jetson Xavier(server)
        '''
        
        self._file_dir = _dir
        self._pose_landmark = _pose
        self._csv_out_dir = _csv_out_dir
        self._iter_num = _iter_num
        
        if self._file_dir is not None and self._iter_num is not None:
            print('_iter_num should be None when _dir is not None')
            return -1
   
        # For reading CSV
        if self._file_dir is not None and self._pose_landmark is None:
           
            # Read csv files
            _keypoints_pd = pd.read_csv(self._file_dir, header = None, delimiter = ",")
                #print('keypoint_pd: \n', _keypoints_pd)
                
            # _frame_len = int(_keypoints_pd.iloc[-1][0])
            _frame_len = int(_keypoints_pd.index[-1]) + 1
            # 프레임 넘버는 열의 개수를 의미하지 않는다...
            
            # Drop the frame feature
            try:
                _keypoints_pd.rename(columns={'0':0}, inplace=True)
            except:
                print('column 0 is already int')
            
            print('keypoints: ', '\n', _keypoints_pd)
            _keypoints_pd = _keypoints_pd.drop([0], axis = 1)
                #print('_keypoints *:', *(_keypoints_pd.head(10)))
            
            """ 기존 csv경로와 같은 구조로 csv파일을 저장하기 위한 경로 설정 """
            # Open csv_out file
            _tmp_start_index = self._file_dir.find('A') # AI... start index
            _tmp_start_indexed = self._file_dir[_tmp_start:]
            _tmp_start_slash = _tmp_start_indexed.find('/') # Find out first '/' after AI...
            _tmp_start_end_index = _tmp_start + _tmp_start_slash
            
            _tmp_end_index = self._file_dir.rfind('/') # Last index of '/' in dir
            _tmp_extra = self._file_dir[_tmp_start_end_index + 1 : _tmp_end_index] # 나머지부분...
 
            _save_dir = os.path.join(self._csv_out_dir, _tmp_extra)
            # ex. '~/saving/2020-jpg/Training/0011_M217M218/0011_M217M218_F_A_10162_10642'
            
            _tmp_file_name = self._file_dir[_tmp_end_index+1: -4] + '_normed.csv' 
            _save_file_name = os.path.join(_save_dir, _tmp_file_name)
                                                            # .csv = exclude last 3 -> -4
                                                            # It is dir, not file_name.csv only
            
            if not os.path.exists(_save_dir):
                os.makedirs(_save_dir)
            
            _csv = open(_save_file_name, 'a', newline = '')
            _csv_writer = csv.writer(_csv, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
    

            for _iter in range(_frame_len):
                print('iter:', '\n', _iter)
                # At every 3 values are x, y, z value of all keypoints. Reshape it.
                _landmarks = np.array(_keypoints_pd.iloc[_iter]).reshape((-1, 3))
                #print('_landmarks', _landmarks, type(_landmarks))
                landmarks = self.Davinci_Normalize(_landmarks)
                # hip에서 멀 수록 xy좌표 값이 커진다. 위로, 오른쪽 관절일수록 음수
                # return = np.array (-1, 3) shape
                
                keypoints_dist = self.Distance_Info(landmarks)
                    #keypoints_dist = keypoints_dist.tolist()
                print('keypoints_dist /n', keypoints_dist)
                keypoints_angle = self.Angle_Info(landmarks)
                    #keypoints_angle = keypoints_angle.tolist()
                
                # np.array가 하나의 리스트로 묶여있음. 각각은 관절부위별 x, y, z임
                keypoints = [_iter]
                keypoints.append(keypoints_dist)
                keypoints.append(keypoints_angle)
                
                print('keypoints: /n', keypoints)
                _csv_writer.writerow(keypoints)
                
            
                
        # For reading pose.landmark(input will be (-1,3) shape already)    
        elif self._file_dir is None and self._pose_landmark is not None:
            """
                인풋이 이미 (-1,3) 리스트 형태로 전달될 것.
                
                아웃풋으로 현재 (-1,3)에 대해 norm, vect 된 리스트/np.array 전달(-이걸로 뭘 할수 있으려나?)
                    또한 csv파일 1분마다? 300프레임마다? 한 개씩 저장 -> jetson전달용
                    그럼, 4~6fps라고 하면, 60초 = 240~360frames.
                    1분이 되었음을 인자로 전달받거나, 이 함수를 몇번 호출했는지 카운트 수(=프레임 수)를 전달받거나.
                    프레임 수를 줄이지 않아도, 저장과 연산 시간에 의해 프레임 수가 알아서 줄지 않을까?
                    일단 300프레임마다 csv 하나 저장하는 것으로 하자. 
            """
            
            _frame_len = 1
            
            ## Check
            print(self._pose_landmark)
            
            # Only make dir and file names when it called every N times
            if self._iter_num == 0:
                
                _tmp_ctime = str(datetime.now())
                # ->'2021-10-12 20:47:25.303356'
                _tmp_file_name = _tmp_ctime[:4]+'_'+_tmp_ctime[5:7]+'_'+_tmp_ctime[8:10]+'_'+_tmp_ctime[11:13]+':'+_tmp_ctime[14:16]+':'+_tmp_ctime[17:19]
                # -> '2021_10_12_20:54:33'
                
                _new_fold_name = "Real_time_prediction"
                _save_dir = os.path.join(self._csv_out_dir, _new_fold_name)
                _save_file_name = os.path.join(self._csv_out_dir, _tmp_file_name)
                
                if not os.path.exists(_save_dir):
                    os.makedirs(_save_dir)
                
                _csv = open(_save_file_name, 'a', newline = '')
                _csv_writer = csv.writer(_csv, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
    
            
            # for _iter in range(_frame_len):
            """
                # Normalize
                # Distance
                # Angle info
                함수 호출 필요!
                + 리턴용 리스트 제작 필요!
            """
            
            # Same part of codes above
            _landmarks = np.array(self._pose_landmark).reshape(-1,3)
            
            landmarks = Davinci_Normalize(_landmarks)

            keypoints_dist = Distance_Info(landmarks)

            keypoints_angle = Angle_Info(landmarks)

            keypoints = [self._iter_num]
            keypoints.append(keypoints_dist)
            keypoints.append(keypoints_angle)


            _csv_writer.writerow(keypoints)

            
            return keypoints # 거리, 각도 리스트 리턴!
            
            
            
        else:
            # error 코드 작성 필요
            print('ERROR (class - Normalize_Landmarks):')
            print('ERROR _dir or _landmark is need. or both were given.')
            return -1

        
class Normalize_Landmarks_2019(Normalize_Landmarks):
    """ Sub class 1 """
    def __init__(self):
        self._keypoint_names = [
            'right_ankle', 'right_knee', 'right_hip',
            'left_hip', 'left_knee', 'left_ankle', 
            'hip',
            'chest',
            'neck', 'head',
            'right_wrist', 'right_elbow', 'right_shoulder',
            'left_shoulder', 'left_elbow', 'left_wrist',
        ]
     
    # 함수화 - Normalize translation and scale
    def Davinci_Normalize(self, landmarks):
        
        _keypoints_loads = landmarks.astype(np.float32)
            # print(_keypoints_loads[0])

        """ Centerize """    
        _left_hip = _keypoints_loads[self._keypoint_names.index('left_hip')]
        _right_hip = _keypoints_loads[self._keypoint_names.index('right_hip')]
        _left_shoulder = _keypoints_loads[self._keypoint_names.index('left_shoulder')]
        _right_shoulder = _keypoints_loads[self._keypoint_names.index('right_shoulder')]

        
        _left_hip_key = _keypoints_loads[_left_hip]
        _right_hip_key = _keypoints_loads[_right_hip]
        _hips = (_left_hip_key + _right_hip_key) * 0.5
        _hips = _hips.astype(np.float32)
            # print(_hips)
        
        _left_shoulder_key = _keypoints_loads[_left_shoulder]
        _right_shoulder_key = _keypoints_loads[_right_shoulder]
        _shoulders = (_left_shoulder_key + _right_shoulder_key) * 0.5
        _shoulders = _shoulders.astype(np.float32)
            # print(_shoulders)

        _body_max_dist = np.max(np.linalg.norm(_keypoints_loads - _hips, axis = 1))

        _torso_size = np.linalg.norm(_shoulders - _hips)
        _torso_size *= 2.5

        _pose_size = max(_torso_size, _body_max_dist)

            # print('org       : ', _iter, ', ', _keypoints_loads[0])
            # print('hips:', _hips, type(_hips[0]))
            # print('keypoints_loads: ', _keypoints_loads, type(_keypoints_loads[0][1]))
        _keypoints_loads -= _hips
            #print('norm_trans: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads /= _pose_size
            #print('norm_scale: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads *= 100 # for easy debug

        return _keypoints_loads


        
class Normalize_Landmarks_2020(Normalize_Landmarks):
    """ Sub class 2 """
    def __init__(self):
        self._keypoint_names = [
            'hip',
            'left_hip', 'left_knee', 'left_ankle',
            'left_foot_index', 'left_little_toe',
            'right_hip', 'right_knee', 'right_ankle',
            'right_foot_index', 'right_littel_toe',
            'back',
            'chest',
            'neck',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'left_thumb_2', 'left_pinky_1',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'right_thumb_2', 'right_pinky_1',
            'nose',
            'left_eye', 'right_eye',
        ]

    # 함수화 - Normalize translation and scale
    def Davinci_Normalize(self, landmarks):
        
        _keypoints_loads = landmarks.astype(np.float32)
            # print(_keypoints_loads[0])

        """ Centerize """    
        _left_hip = _keypoints_loads[self._keypoint_names.index('left_hip')]
        _right_hip = _keypoints_loads[self._keypoint_names.index('right_hip')]
        _left_shoulder = _keypoints_loads[self._keypoint_names.index('left_shoulder')]
        _right_shoulder = _keypoints_loads[self._keypoint_names.index('right_shoulder')]
        print('left_hip', _left_hip)
        
        _hips = (_left_hip + _right_hip) * 0.5
        _hips = _hips.astype(np.float32)
            # print(_hips)

        _shoulders = (_left_shoulder + _right_shoulder) * 0.5
        _shoulders = _shoulders.astype(np.float32)
            # print(_shoulders)

        _body_max_dist = np.max(np.linalg.norm(_keypoints_loads - _hips, axis = 1))

        _torso_size = np.linalg.norm(_shoulders - _hips)
        _torso_size *= 2.5

        _pose_size = max(_torso_size, _body_max_dist)

            # print('org       : ', _iter, ', ', _keypoints_loads[0])
            # print('hips:', _hips, type(_hips[0]))
            # print('keypoints_loads: ', _keypoints_loads, type(_keypoints_loads[0][1]))
        _keypoints_loads -= _hips
            #print('norm_trans: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads /= _pose_size
            #print('norm_scale: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads *= 100 # for easy debug

        return _keypoints_loads
    
    
    
        
class Normalize_Landmarks_mediapipe(Normalize_Landmarks):
    """ Sub class 3 """
    def __init(self):
        print('sub class - mediapipe init')
        super(Normalize_Landmarks_mediapipe, self).__init__()
    
    def Davinci_Normalize(self, landmarks):
        
        _keypoints_loads = landmarks.astype(np.float32)
            # print(_keypoints_loads[0])

        """ Centerize """    
        _left_hip = _keypoints_loads[self._keypoint_names.index('left_hip')]
        _right_hip = _keypoints_loads[self._keypoint_names.index('right_hip')]
        _left_shoulder = _keypoints_loads[self._keypoint_names.index('left_shoulder')]
        _right_shoulder = _keypoints_loads[self._keypoint_names.index('right_shoulder')]
        print('left_hip', _left_hip)
        
        _hips = (_left_hip + _right_hip) * 0.5
        _hips = _hips.astype(np.float32)
            # print(_hips)

        _shoulders = (_left_shoulder + _right_shoulder) * 0.5
        _shoulders = _shoulders.astype(np.float32)
            # print(_shoulders)

        _body_max_dist = np.max(np.linalg.norm(_keypoints_loads - _hips, axis = 1))

        _torso_size = np.linalg.norm(_shoulders - _hips)
        _torso_size *= 2.5

        _pose_size = max(_torso_size, _body_max_dist)

            # print('org       : ', _iter, ', ', _keypoints_loads[0])
            # print('hips:', _hips, type(_hips[0]))
            # print('keypoints_loads: ', _keypoints_loads, type(_keypoints_loads[0][1]))
        _keypoints_loads -= _hips
            #print('norm_trans: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads /= _pose_size
            #print('norm_scale: ', _iter, ', ', _keypoints_loads[0])
        _keypoints_loads *= 100 # for easy debug

        return _keypoints_loads
