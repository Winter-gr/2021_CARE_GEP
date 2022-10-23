'''
    2020 data는 모듈화 진행 X
    input: 
        1. json_folder_dir
            Ex. "../../AI_hub_second/2020/Validation/"
            Ex. "../../AI_hub_second/2020/Training/"
        2. csv_out_dir
            Ex. "../../AI_hub_second/Json_to_csv_module_TEST/2020/Validation"
            Ex. "../../AI_hub_second/Json_to_csv_module_TEST/2020/Training"
        
'''


import os
import csv
import time
import json

'''
# 경로를 일단 img_dir 처럼 임의로 지정해두면, 리스트 생성
img_dir = "../../AI_hub_second/0011_M217M218/0011_M217M218_F_A_10162_10642"
img_file = [fold for fold in os.listdir(img_dir)]
img_file = sorted(img_file)
'''

def Json_to_CSV_2020_func(json_folder_dir, csv_out_dir):

    # json 데이터가 저장된 제일 상위 디렉토리를 전달한다. Training, Validation 까지는 적어서 넘긴다고 가정.
    json_folder_dir = json_folder_dir

    # Training 디렉토리 바로 아래에 폴더들을 리스트화한다.
    # folder 리스트. 118개 있음. 이 안에서 Front 같은것만 추출해야 함.
    json_folder_list = [fold for fold in os.listdir(json_folder_dir)] 
    json_folder_list = sorted(json_folder_list)

    # csv 파일이 저장될 경로를 받아온다.
    csv_out_dir = csv_out_dir

    folder_to_remove_list = []

    for folder in json_folder_list:

        """ 1. Make Path """
        keypoints_only_flag = False

        # 현재 for문이 가리키고 있는 폴더명을 저장한다.
        folder_name = folder
        json_file_dir = os.path.join(json_folder_dir, folder_name) # 확장자 없이, / 도 없이 경로로 생성됨.

        # Each folder has Front, Left, Right camera angles. Among these, only choose Front folder
        _front_cam_folder = [fold for fold in os.listdir(json_file_dir) if fold.find("F") != -1] # ['0012_M273M274_F_B_11123_11603']
        _front_cam_folder = _front_cam_folder[0]

        # Then, make relative path to exact json files each from this python file
        json_list = sorted(os.listdir(os.path.join(json_file_dir, _front_cam_folder))) # it is just a path

        # Make directories for saving csv
        save_dir = os.path.join(csv_out_dir, folder_name, _front_cam_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # json file list for json_list dir
        json_file_list = []
        json_end_each = json_list[len(json_list) -1]



        """ 2. Find out .jpg.json files to decide how to extract information """
        for json_each in json_list:

            # -1 returns if it cannot find ".jpg" in each json file names
            if json_each.find(".jpg") == -1:
                json_file_list.append(json_each)



        '''    
            if json_each == json_end_each:
                if len(json_file_list) == 0:
            # print(json_list[len(json_list)-1])
            # 파일의 끝에 도달했는데 json_file_list가 비었다면 .jpg.json만 존재했던 것이므로, 리스트 삭제 필요
            #if json_each == json_end_each:

        # print('all:', len(json_list))    
        '''


        """ 3. keypoints saving setting  - 공통사항 """
        keypoint_1 = os.path.join(save_dir, _front_cam_folder) + "_1.csv"
        keypoint_2 = os.path.join(save_dir, _front_cam_folder) + "_2.csv"

        csv_out_k_1 = open(keypoint_1, 'a', newline='')
        csv_out_keypoint_1 = csv.writer(csv_out_k_1, delimiter = ',')
        csv_out_k_2 = open(keypoint_2, 'a', newline='')
        csv_out_keypoint_2 = csv.writer(csv_out_k_2, delimiter = ',')



        """ 3-1. Keypoints and bbox extracting """
        jpg_test = len(json_file_list)

        # Only .jpg.json files exist? no = run this
        if jpg_test != 0:
            # print(json_file_list)
            # print('len:', len(json_file_list))

            # Bbox csv setting
            bbox_1 = os.path.join(save_dir, _front_cam_folder) + "_bbox1.csv"
            bbox_2 = os.path.join(save_dir, _front_cam_folder) + "_bbox2.csv"

            csv_out_b_1 = open(bbox_1, 'a', newline='')
            csv_out_bbox_1 = csv.writer(csv_out_b_1, delimiter = ',')
            csv_out_b_2 = open(bbox_2, 'a', newline='')
            csv_out_bbox_2 = csv.writer(csv_out_b_2, delimiter = ',')


            # json_file_list only contains .json file. (no .jpg.json)
            for json_each in json_file_list:
                # print(json_each)

                # Variable Initializing
                json_keypoint_1 = []
                json_keypoint_2 = []
                json_bbox_1 = []
                json_bbox_2 = []

                json_dir = os.path.join(json_file_dir, _front_cam_folder, json_each)

                with open(json_dir, 'r') as file:
                    contents = file.read()
                    json_data = json.loads(contents)


                # 0012_M403M404_F_A_11118_11598_0000000.jpg.json
                frame_index = int(json_data["images"][0]["frame_index"])

                json_keypoint_1 = [frame_index]
                json_keypoint_2 = [frame_index]
                json_bbox_1 = [frame_index]
                json_bbox_2 = [frame_index]

                ''' 발작의 흔적
                if json_each[:13] == "0012_M403M404":
                    f_name = os.path.join(save_dir, _front_cam_folder) + "_ffff.csv"
                    fucking_idiot = open(f_name, 'w', newline='')
                    fucking_write = csv.writer(fucking_idiot, delimiter = ',')



                    json_bbox_1 = [frame_index]
                    json_bbox_2 = [frame_index]
                    bull_shit = [1]

                    for k in range(4):
                        json_bbox_1.append(int(json_data["annotations"][0]["bbox"][k]))
                        json_bbox_2.append(int(json_data["annotations"][1]["bbox"][i]))
                        bull_shit.append('Fuck')
                        bull_shit.append('shit')

                    print(json_bbox_1)

                    fucking_write.writerow(bull_shit)
                    csv_out_writer_b_1.writerow(json_bbox_1)
                    csv_out_writer_b_2.writerow(json_bbox_2)

                    #    print('bbox: ', int(json_data["annotations"][0]["bbox"][k]))
                ''' 

                for i in range(4):
                    bbox_1_contents = int(json_data["annotations"][0]["bbox"][i])
                    bbox_2_contents = int(json_data["annotations"][1]["bbox"][i])

                    json_bbox_1.append(bbox_1_contents)          
                    json_bbox_2.append(bbox_2_contents) 

                csv_out_bbox_1.writerow(json_bbox_1)
                csv_out_bbox_2.writerow(json_bbox_2)

                for j in range(81):
                    _j = j + 1
                    if _j%3 == 1:
                        _value_tmp_1 = round(((json_data["annotations"][0]['keypoints'][j])/1920), 6)
                        _value_tmp_2 = round(((json_data["annotations"][1]['keypoints'][j])/1920), 6)
                    elif _j%3 == 2:
                        _value_tmp_1 = round(((json_data["annotations"][0]['keypoints'][j])/1080), 6)
                        _value_tmp_2 = round(((json_data["annotations"][1]['keypoints'][j])/1080), 6)
                    else:
                        _value_tmp_1 = None
                        _value_tmp_2 = None
                    json_keypoint_1.append(_value_tmp_1)
                    json_keypoint_2.append(_value_tmp_2)

                csv_out_keypoint_1.writerow(json_keypoint_1)
                csv_out_keypoint_2.writerow(json_keypoint_2)



            '''
                ###########.jpg.json으로만 이루어져있는 폴더는 keypoint만 추출

                # .jpg.json으로만 이루어져있는 폴더들은 따로 표시 - bbox정보 없음.
                # if len(json_file_list) == 0:
                    # 폴더이름만 저장
                    # 파일 이름 예시: 0163_M532M533_F_B_12742_13302_0000170.jpg

                    # print(folder_to_remove)


                    # keypoints_only_flag = True

                    # print('wow', json_end_each)
                # print('to remove:', len(folder_to_remove_list))

                # print(len(folder_to_remove_list)) # 72개나 나옴...!

                # 이 else에 왔다 == 모두 jpg 확장자가 포함되었다
                # 이 때의 json_list에는 모두 .jpg.json이다
            '''


            """ 3-2. Only Keypoints extracting """
        # this time, json_list only contains .jpg.json files
        # else:
        elif jpg_test == 0:
            '''
            # folder_to_remove = json_end_each[:13]
            # folder_to_remove_list.append(folder_to_remove)
            # json_jpg_list = json_list
            # print('jpg:', json_file_dir)
            '''

            for jpg_each in json_list:

                # Variable Initializing
                jpg_keypoint_1 = []
                jpg_keypoint_2 = []

                json_jpg_dir = os.path.join(json_file_dir, _front_cam_folder, jpg_each)

                with open(json_jpg_dir, 'r') as file:
                    contents_jpg = file.read()
                    json_jpg_data = json.loads(contents_jpg)

                frame_index_jpg = int(jpg_each[-12:-9])
                # frame_index = json_data["images"][0]["frame_index"]

                jpg_keypoint_1 = [frame_index_jpg]
                jpg_keypoint_2 = [frame_index_jpg]


                for j in range(25):
                    jpg_keypoint_1.append(round(((json_jpg_data["annotations"][0]['keypoints'][str(j)][0])/1920), 6))
                    jpg_keypoint_1.append(round(((json_jpg_data["annotations"][0]['keypoints'][str(j)][1])/1080), 6))
                    jpg_keypoint_1.append(None)

                    jpg_keypoint_2.append(round(((json_jpg_data["annotations"][1]['keypoints'][str(j)][0])/1980), 6))
                    jpg_keypoint_2.append(round(((json_jpg_data["annotations"][1]['keypoints'][str(j)][1])/1080), 6))
                    jpg_keypoint_2.append(None)

                csv_out_keypoint_1.writerow(jpg_keypoint_1)
                csv_out_keypoint_2.writerow(jpg_keypoint_2)

            # print('jpg end')

