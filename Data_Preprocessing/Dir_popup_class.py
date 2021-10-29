'''
    인풋:
        1. 상위 디렉토리
        2. 파일까지의 직접적인 디렉토리(코드가 접근해야하는. 예시로 입력)
        3. 파일 확장자(모듈 실행시 구분을 위함)
        
    아웃풋:
        1. 처음 동작: 리스트 제작 후 첫 번째 파일 디렉토리 popup
        2. 이후 동작: 파일 디렉토리 pop up
'''

'''

'../../AI_hub_second/video_action_3'
'../../AI_hub_second/video_action_3/3-1/3-1_001-C01.mp4'
OR
'../../AI_hub_second/2020-jpg'
'../../AI_hub_second/2020-jpg/Training/0011_M217M218/0011_M217M218_F_A_10162_10642/0011_M217M218_F_A_10162_10642_0000000.jpg'

OR
'../../AI_hub_second/2020'
'../../AI_hub_second/2020/Training/0011_M217M218/0011_M217M218_F_A_10162_10642/0011_M217M218_F_A_10162_10642_0000000.json'

OR
'../../AI_hub_second/2020/Annotation_2D_tar/2D-1'
'../../AI_hub_second/2020/Annotation_2D_tar/2D-1/2-1/2-1_001-C01_2D.json'

OR
'../../AI_hub_second/video_action_3/'
'../../AI_hub_second/video_action_3/3-1/3-1_001-C01.mp4'

'''

import os
import numpy as np
import time


class Directory_popup(object):
    def __init__(self, in_dir, ex_dir, target_format):
        print('Directory_popup Class init')        
        self._in_dir = in_dir
        self._ex_dir = ex_dir
        self._target_format = target_format
        self._target_list = []
        
        if self._ex_dir.find('json') != -1:
            self._json_flag = True
        else:
            self._json_flag = False
        
        # Count '/', which means how deep it needs to dig in
        _len_dir  = len(self._in_dir)
        _diff_dir = self._ex_dir[_len_dir:]
        
        _count = _diff_dir.count('/')
        # 2019 data ver -> _count = 2
        # 2020 data ver -> _count = 4
        
        if _count == 2: # 3-1 3-2 3-3... 으로 폴더가 바로 나타나는 경우(2019)
            _dir_1 = os.listdir(self._in_dir)
            _folder_list = sorted(_dir_1)
            #print('1:' ,_folder_list)
            
            for fold in _folder_list:
                _dir_2 = sorted(os.listdir(os.path.join(self._in_dir, fold)))
                #print('2: ', _dir_2)
                
                for _each in _dir_2:
                    self._target_list.append(os.path.join(self._in_dir, fold, _each))
                    #print('f:', self._target_list)
                    time.sleep(0.1)
            
        elif _count == 4:
            _dir_1 = os.listdir(self._in_dir)
            
            for i in _dir_1: # Training or Validation
                _target_1 = i

                _dir_2 = sorted(os.listdir(os.path.join(self._in_dir, _target_1)))
                # print('2: ', _dir_2) # '0149_M416M417 ... '
                
                for M_folder in _dir_2:
                    # Contains M000M000 folder names
                    _dir_3 = sorted(os.listdir(os.path.join(self._in_dir, _target_1, M_folder)))
                    
                    # print('3: ', _dir_3) # '0149_M416M417_L_B_17542_18164 ...'
                    
                    for FLR_folder in _dir_3:
                        if FLR_folder.find('F') != -1: # if there is no F then return -1

                            _dir_4 = sorted(os.listdir(os.path.join(self._in_dir, _target_1, M_folder, FLR_folder)))
                            # print('4: ', _dir_4)
                            
                            _for4_count_max = len(_dir_4)
                            # print('len dir4: ', _for4_count_max, 'in :', _dir_3)
                            
                            _for4_count = 0
                            _count_json = 0
                            
                            for _each in _dir_4:
                                _for4_count = _for4_count + 1
                                
                                # if it is Json file dir, needs to extract .jpg.json
                                if self._json_flag:
                                    if _each.find('jpg') == -1:
                                        _count_json = _count_json + 1
                                        self._target_list.append(os.path.join(self._in_dir, _target_1, M_folder, FLR_folder, _each))
                                        # print('no jpg and json!', _each)
                                else:
                                    self._target_list.append(os.path.join(self._in_dir, _target_1, M_folder, FLR_folder, _each))
                                
                                # In this case, there is only .jpg.json -> so just append all.
                                if _for4_count == _for4_count_max and _count_json == 0:
                                    
                                    for _each in _dir_4:
                                        self._target_list.append(os.path.join(self._in_dir, _target_1, M_folder, FLR_folder, _each))
                                        # print('only jpgjson!', _each)
                                
                                
                                
        # print(self._target_list[4810])
        # print(_count)  
        
        self._target_len = len(self._target_list)
        self._target_list.reverse()
        
        
        
    def Dir_Pop_Up(self):
        new_dir = self._target_list.pop()
        
        return new_dir

    def Dir_Length(self):
        return self._target_len
