# Import modules
import os
import json
import csv
import numpy as np
import time

# Import DIR POP UP module(in the same dir)
import Dir_popup_class as pop
import Json_to_CSV_2020_func as to_csv_2020

'''
    input
        1. 파일들을 포함하는 상위 디렉토리
            ex. _dir_1 = '../../AI_hub_second/2020'
        2. 파일까지의 직접적인 디렉토리(코드가 접근해야하는. 예시로 입력)
            ex. _dir_2 = '../../AI_hub_second/2020/Training/0011_M217M218/
                            0011_M217M218_F_A_10162_10642/0011_M217M218_F_A_10162_10642_0000000.json'
        3. 파일 확장자(모듈 실행시 구분을 위함)

    output
        1. 2019: bbox: 좌 상단 x,y 우 하단 x,y 가 리턴됨.

'''

class InValidException(Exception):
    def __init__(self):
        super(InValidException, self).__init__("Class Initialize Failed")


class Json_to_CSV(object):
    ''' PARENT CLASS '''
    def __init__(self, fold_dir, file_dir, csv_dir, file_format, year):
        print('Json_to_CSV 생성자 호출')
        self._fold_dir = fold_dir
        self._file_dir = file_dir
        self._csv_dir = csv_dir
        
        self._same_dir = 'AI_hub_second'
        self._same_dir_len = len(self._same_dir)
        self._file_format = file_format
        
        self._data_year = int(year)
        # 예외처리
        if self._data_year != 2019 and self._data_year != 2020:
            print("ERROR: json to csv conversion is valid only for 2019 or 2020 data ver")
            raise InValidException

        if self._data_year == 2019:
            # Dir Popup class Initialize
            self.Dir_popup = pop.Directory_popup(self._fold_dir, self._file_dir, self._file_format)
            self._iter_max = self.Dir_popup.Dir_Length()
    
    
    def Start_Conversion(self):
        
        if self._data_year == 2019:
            self.Start_Conversion_2019()
        else:
            self.Start_Conversion_2020()
        
    
    def Start_Conversion_2019(self):
        # '../../AI_hub_second/Annotation_2D_tar/2D-1/2-1' 가정.
        # '../../AI_hub_second/Annotation_2D_tar/2D-1/2-10' 도 포함할 수 있어야 함.
        # '../../AI_hub_second/Json_to_csv_module_TEST/' : 저장 dir
        
        for _iter in range(self._iter_max):          
            json_file_dir = self.Dir_popup.Dir_Pop_Up()
            
            if _iter == 0:
                _tmp_csv_dir = 'Annotation_2D_tar/2D-1'
                save_csv_dir = os.path.join(self._csv_dir, _tmp_csv_dir) 
                # save_csv_dir = custom csv dir + 'Annotation_2D_tar/2D-1'
                # Ex. ../../AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1'
            
            # Extract 2-1... 2-10... for mkdir
            _tmp_dir_iter_len = len('../../AI_hub_second/Annotation_2D_tar/2D-1')
            _csv_name = json_file_dir[_tmp_dir_iter_len +1:] # 2-10/2-10_C01_xxxx.json
            _last_slash = _csv_name.find('/')
            _tmp_dir_iter = _csv_name[:_last_slash]# == 2-1 ... 2-10...
            
            _csv_folder = os.path.join(save_csv_dir, _tmp_dir_iter)
            # Ex. '../../AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1/2-12'
            
            if not os.path.exists(_csv_folder):
                os.makedirs(_csv_folder)

                
            # Loads json data
            with open(json_file_dir, 'r') as json_org:
                _contents = json_org.read()
                _json_data = json.loads(_contents)
            
            # Caculate length of json frames
            length_json_frame = len(_json_data["annotations"])
            img_frame = len(_json_data["images"])
            
            
            # For making new csv file
            open_csv_dir = os.path.join(save_csv_dir, _csv_name)
            # Ex. '../../AI_hub_second/Json_to_csv_module_TEST/Annotation_2D_tar/2D-1/2-12/2-12_C01_xxxx.json
            
            # Join the path: for csv files
            _csv_keypoint = open_csv_dir[:-5] + "_key.csv"
            _csv_bbox = open_csv_dir[:-5] + "_bbox.csv"
            
            csv_out_key = open(_csv_keypoint, 'w', newline='')
            csv_out_writer_key = csv.writer(csv_out_key, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
            csv_out_bbox = open(_csv_bbox, 'a', newline='')
            csv_out_writer_bbox = csv.writer(csv_out_bbox, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
            
            
            
            # Iteration for each json files
            for i in range(length_json_frame):
                # Initialize Variables
                _json_keypoints = []
                _json_bbox = []
                
                if length_json_frame == img_frame:
                    # 아 씨발 2019 데이터 세트중에서 몇개는 images 정보는 14개만 해놓고 keypoint는 25개로 그지같이 인덱싱해놨네
                    frame_number = _json_data["images"][i]["img_path"]
                    frame_number = int(frame_number[-7:-4])
                else:
                    frame_number = int(10 * i + 9)
                    
                _json_keypoints = [frame_number]
                _json_bbox = [frame_number]
                
                # Keypoints slicing
                _keypoints_tmp = []
                _keypoints_size = len(_json_data["annotations"][i]['keypoints'])
                # Appending annotations
                for k in range(_keypoints_size):
                    
                    _k = k+1
                    if _k%3 == 1:
                        _value_tmp = round(((_json_data["annotations"][i]['keypoints'][k])/1920),6)
                        _json_keypoints.append(_value_tmp)
                    elif _k%3 == 2:
                        _value_tmp = round(((_json_data["annotations"][i]['keypoints'][k])/1080),6)
                        _json_keypoints.append(_value_tmp)
                    else:
                        _json_keypoints.append(None)
                        # np.nan 이란 값을 넣으면 리스트로 write된다(아무것도 없는 값은 열에 못 넣는가벼)
                csv_out_writer_key.writerow(_json_keypoints)

                
                _bbox_size = len(_json_data["annotations"][i]['bbox'])
                for j in range(_bbox_size):
                    _json_bbox.append(_json_data["annotations"][i]['bbox'][j])
                    # 좌 상단 x,y 우 하단 x,y 가 저장되어, 나중에 crop하고 싶다면, 행렬에 그냥 이 값들을 넣으면 됨!
                
                csv_out_writer_bbox.writerow(_json_bbox)
                    
                    
                    
    def Start_Conversion_2020(self):
        # fold_dir = '../../AI_hub_second/2020'
        # file_dir = '../../AI_hub_second/2020/Training/0011_M217M218/0011_M217M218_F_A_10162_10642'
        # csv_dir = '../../AI_hub_second/Json_to_csv_module_TEST/'

        _json_dir = self._fold_dir
        _csv_dir = self._csv_dir
        to_csv_2020.Json_to_CSV_2020_func(json_folder_dir = _json_dir, csv_out_dir = _csv_dir)




""" 동작 예시 - # 모든 dataset들은 AI~ 로 시작하는 폴더 안에서 정렬되어 있다고 가정(..../AI~~~/2020data 또는 ..../AI~~~~/2019data 이런식
    
    ### 2019
    
    fold_dir = '../../AI_hub_second/Annotation_2D_tar/2D-1'
    file_dir = '../../AI_hub_second/Annotation_2D_tar/2D-1/2-1'
    csv_dir = '../../AI_hub_second/Json_to_csv_module_TEST/'
    file_format = '.json'
    data_year = 2019

    json_to_csv = Json_to_CSV(fold_dir = fold_dir, file_dir = file_dir, csv_dir = csv_dir, 
                              file_format = file_format, year=data_year)
    json_to_csv.Start_Conversion()


    ### 2020 - Validation

    fold_dir_v = "../../AI_hub_second/2020/Validation/"
    file_dir = None
    csv_dir_v = "../../AI_hub_second/Json_to_csv_module_TEST/2020/Validation"
    file_format = '.json'
    data_year = 2020

    json_to_csv = Json_to_CSV(fold_dir = fold_dir_v, file_dir = file_dir, csv_dir = csv_dir_v, 
                              file_format = file_format, year=data_year)
    json_to_csv.Start_Conversion()

    ### 2020 - Training

    fold_dir_v = "../../AI_hub_second/2020/Training/"
    file_dir = None
    csv_dir_v = "../../AI_hub_second/Json_to_csv_module_TEST/2020/Training"
    file_format = '.json'
    data_year = 2020

    json_to_csv = Json_to_CSV(fold_dir = fold_dir_v, file_dir = file_dir, csv_dir = csv_dir_v, 
                              file_format = file_format, year=data_year)
    json_to_csv.Start_Conversion()

"""
