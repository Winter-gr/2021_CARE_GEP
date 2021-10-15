import Dir_popup_class as popup
import Video_to_CSV as video

# Dir setting
mp4_fold_dir = "../../AI_hub_second/video_action_3"
mp4_example_dir = "../../AI_hub_second/video_action_3/3-3"
csv_out_dir = "../../AI_hub_second/Json_to_csv_module_TEST"


# Initialize Dir popup class
_Dir_inst = popup.Directory_popup(in_dir = mp4_fold_dir, ex_dir = mp4_example_dir, target_format = 'mp4')
_Dir_len = _Dir_inst.Dir_Length()


# Iteration for each mp4 files
for _iter in range(_Dir_len):
    # Extract path for mp4 each
    mp4_dir = _Dir_inst.Dir_Pop_Up()
    
    # Extract skeleton
    Video_to_csv_TEST = video.ConvertToCSV(csv_out_dir, 'mp4', verbose = False)
    exit_value = Video_to_csv_TEST.extract_skeleton(mp4_dir)
    print(exit_value)
    if exit_value == -1:
        break
    
