import cv2
import os
import pandas as pd
from collections import defaultdict, Counter
from typing import List


def read_label(label_path, label_map_path):
    """
    라벨 CSV와 라벨 맵을 읽어 필요한 컬럼 데이터를 반환.
    """
    act_level = label_map_path.split(os.sep)[-1].split('_')[-1].split('.')[0]
    
    label = pd.read_csv(label_path)
    label_map = pd.read_csv(label_map_path)
    label_idx_list = label_map['index'].tolist()
    label_list = label_map[f'{act_level}_activity'].tolist()
    
    file_ids = label['file_id'].tolist()
    start_frames = label['frame_start'].tolist()
    end_frames = label['frame_end'].tolist()
    activities = label['activity'].tolist()
    
    return file_ids, start_frames, end_frames, activities, label_idx_list, label_list


def make_label_map(input_path, output_path):
    """
    입력 CSV 파일로부터 라벨 맵을 생성하고 저장.
    """
    # CSV 파일 읽기
    label_map = pd.read_csv(input_path)
    activities = Counter(label_map['activity'])
    act_level = input_path.split(os.sep)[-1].split('.')[0]
    
    # 라벨 맵 딕셔너리 생성
    label_map_dict = defaultdict(list)
    for idx, key in enumerate(activities.keys()):
        label_map_dict['index'].append(idx)
        label_map_dict[f'{act_level}_activity'].append(key)
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory created: {output_path}")
    else:
        print(f"Directory already exists: {output_path}")
    
    # 출력 파일 경로 설정
    output_file = os.path.join(output_path, f"label_map_{input_path.split(os.sep)[-2]}_{act_level}.csv")
    pd.DataFrame(label_map_dict).to_csv(output_file, index=False)
    
    print(f"Label map saved to: {output_file}")
    return label_map_dict, output_file


def make_new_label(file_id, activity, label_idx_list, label_list, cumulative_csv_path, i):
    """
    VP 디렉토리 안의 누적 CSV 파일에 라벨 정보를 추가.
    """
    # 라벨 인덱스 찾기
    mapping = zip(label_idx_list, label_list)
    idx = next((x for x, y in mapping if activity == y), None)

    if idx is None:
        print(f"Error: Activity '{activity}' not found in label list")
        return

    # 새로운 라벨 정보 생성
    new_label = {'file_id': file_id.split('.')[0] + f'_cut_{i}.mp4', 'activity': activity, 'label_idx': idx}
    new_label_df = pd.DataFrame(new_label, index=[0])

    # 누적 CSV 파일에 데이터 추가 (append 모드)
    if not os.path.exists(cumulative_csv_path):
        # 파일이 없으면 헤더와 함께 저장
        new_label_df.to_csv(cumulative_csv_path, index=False)
    else:
        # 파일이 있으면 헤더 없이 추가
        new_label_df.to_csv(cumulative_csv_path, index=False, mode='a', header=False)

    print(f"Label for file_id '{file_id}' saved to {cumulative_csv_path}")


def cut_video_by_frames(input_video_path, output_video_path, start_frame, end_frame, file_id):
    """
    비디오를 주어진 프레임 범위로 잘라서 저장.
    """
   
    # Open the video
    cap = cv2.VideoCapture(input_video_path + '.mp4')
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if frame range is valid
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print(f"Error: Invalid frame range for video {input_video_path}")
        cap.release()
        return None
    
    if end_frame - start_frame < 16:
        print(f"Skipping video {file_id}: frame range too short ({end_frame - start_frame} frames)")
        cap.release()
        return None
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Loop through frames and write to output
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()
    print(f"Video cut successfully and saved to {output_video_path}")
    
    return output_video_path


if __name__ == '__main__':
    root_dir = 'C:\\Final_project\\cut_videos\\steering_wheel'  # VP 디렉토리가 생성될 최상위 폴더
    label_map_dir = 'C:\\Final_project\\label_maps'
    input_video_dir = 'C:\\Final_project\\data\\steering_wheel'  # 입력 비디오 디렉토리 경로

    # Read label information from CSV file
    label_paths = [
        'C:\\Final_project\\data\\activities_3s\\steering_wheel\\midlevel.chunks_90.csv',
        'C:\\Final_project\\data\\activities_3s\\steering_wheel\\objectlevel.chunks_90.csv',
        'C:\\Final_project\\data\\activities_3s\\steering_wheel\\tasklevel.chunks_90.csv'
    ]

    label_map_paths = []
    for label_path in label_paths:
        _, label_map_path = make_label_map(label_path, label_map_dir)
        label_map_paths.append(label_map_path)

    file_ids, start_frames, end_frames, activities, label_idx_list, label_list = read_label(label_paths[0], label_map_paths[0])

    begin_time = time()
    # Loop through each entry in the label data
    for i in range(len(start_frames)):
        file_id = file_ids[i]  # Example: 'vp1/run1b_2018-05-29-14-02-47.kinect_color'
        activity = activities[i]
        start_frame = start_frames[i]
        end_frame = end_frames[i]
        
        # Generate input video path (no additional join with file_id)
        input_video_path = os.path.join(input_video_dir, file_id)

        # Extract VP folder name and output directory
        vp = file_id.split('/')[0]  # Example: 'vp1'
        vp_output_dir = os.path.join(root_dir, vp)
        os.makedirs(vp_output_dir, exist_ok=True)

        # Generate output video path
        output_video_path = os.path.join(
            vp_output_dir,
            os.path.splitext(os.path.basename(file_id))[0] + f'_cut_{i}.mp4'
        )

        # Cut video by frames
        cut_video_result = cut_video_by_frames(input_video_path, output_video_path, start_frame, end_frame, os.path.basename(file_id))

        # Check if the video was successfully cut before updating the label CSV
        if cut_video_result:  # If the result is not None
            cumulative_csv_path = os.path.join(vp_output_dir, f"{vp}_label.csv")
            make_new_label(file_id, activity, label_idx_list, label_list, cumulative_csv_path, i)
    
    end_time = time()
    
    print(f"{round(end_time - begin_time, 3)} sec")

