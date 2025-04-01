import pandas as pd
import os
import json
import subprocess
import time

def trim_video(input_video_path, output_video_path, start_frame, end_frame, fps):
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time

    # FFmpeg 명령어 생성
    ffmpeg_command = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", input_video_path,                # 입력 파일
        "-ss", f"{start_time}",            # 시작 시간
        "-t", f"{duration}",               # 지속 시간
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        output_video_path                      # 출력 파일
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Successfully trimmed video: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while trimming video: {e}")



if __name__ == '__main__':
    kinetic_path= r'C:\Final_project\cut_videos\kinect_color'
    csv_path= r'C:\Final_project\data\activities_3s\kinect_color'
    all_anno_json = 'activity_labels.json'

    vol0=["midlevel.chunks_90.split_0.test","midlevel.chunks_90.split_0.train","midlevel.chunks_90.split_0.val"]
    vol1=["midlevel.chunks_90.split_1.test","midlevel.chunks_90.split_1.train","midlevel.chunks_90.split_1.val"]
    vol2=["midlevel.chunks_90.split_2.test","midlevel.chunks_90.split_2.train","midlevel.chunks_90.split_2.val"]

    vol_list=[vol0,vol1,vol2]
    str_vol_list=['vol0','vol1','vol2']
    data_set=['test','train','val']


    cur=os.getcwd()
    cur=cur.replace('\\','/')

    with open(all_anno_json, "r") as json_file:
        act_dict = json.load(json_file)

    for idx_,k in enumerate(vol_list):
        for idx,i in enumerate(data_set):
            new_data = []
            added_path=str(cur)+'/'+str_vol_list[idx_]+f'_{i}'

            if not(os.path.isdir(added_path)):
                os.mkdir(i)
            count = 0


            st_=time.time()
            input_csv=csv_path+k[idx]+'.csv'
            df = pd.read_csv(input_csv)
            filtered_df = df[df["frame_end"] - df["frame_start"] > 16]

            for _, row in filtered_df.iterrows():
                clip_length = row["frame_end"] - row["frame_start"]
                activity = row["activity"]
                label = act_dict.get(activity, "Unknown")
                output_file=f"[{count}]{row['file_id'].replace('/','_')}"
                new_data.append({
                    "data_index": count,
                    "file_id": output_file,
                    "clip_length": clip_length,
                    "label": label,
                    "activity": activity,
                })
                fps=15
                trim_video((kinetic_path+row["file_id"]+".mp4"), (added_path+'/'+output_file+".mp4"), row["frame_start"], row["frame_end"], fps)
                count += 1

                ed_=time.time()
                print("spent secs//",ed_-st_)
                print("count//",count)

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(added_path+f"/{i}.csv", index=False)