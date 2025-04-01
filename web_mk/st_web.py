import streamlit as st
from PIL import Image
import os
import base64
from dotenv import load_dotenv

import tempfile # 임시 디렉토리와 파일 생성하기 위해 사용
from LLMtoTTS import Play_LLM, play_tts

import io # 메모리에 데이터를 저장하고 부르기 위해 사용
import av # 오디오, 비디오 스트림을 처리하기 위해 사용
import numpy as np
import cv2
from whole_pipeline import final_pipeline

load_dotenv()

# Streamlit static directory 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')

# 첫 번째 페이지의 배경 GIF 설정
background_gif = os.path.join(static_dir, 'background.gif')

video_clip_path=r'.\static\uploaded_video.mp4'
label_map_csv_path=r'.\label_map_midlevel.csv'

# 이미지 불러오기
# 이미지 base64 인코딩
if os.path.exists(background_gif):
    bg_image_base64 = base64.b64encode(open(background_gif, "rb").read()).decode()
else:
    st.error("배경 이미지 파일을 찾을 수 없습니다.")

# Streamlit에서 배경 이미지와 함께 콘텐츠 렌더링하기
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/gif;base64,{bg_image_base64}');
        background-size: cover;
        background-position: center;
    }}
    .main-content {{
        position: relative;
        z-index: 1;
        color: white;
        text-align: center;
        padding-top: 20px;
        background: rgba(0, 0, 0, 0.6); /* 불투명 배경 추가 */
        border-radius: 10px;
        padding: 20px;
        display: inline-block;
    }}
    .custom-title {{
        color: #FFD700; /* 원하는 색상 (금색) */
        font-size: 1.8em; /* 글자 크기 */
        font-weight: bold; /* 굵은 글자 */
        text-align: center; /* 가운데 정렬 */
        margin-top: 20px;
        background: rgba(0, 0, 0, 0.6); /* 불투명 배경 추가 */
        border-radius: 10px;
        padding: 10px;
        display: inline-block;
    }}
    .custom-upload {{
        margin-top: 20px;
        background: rgba(0, 0, 0, 0.6); /* 불투명 배경 추가 */
        border-radius: 10px;
        padding: 10px;
        display: inline-block;
    }}
    .custom-success {{
        margin-top: 20px;
        background: rgba(0, 100, 0, 0.6); /* 성공 메시지 배경색 */
        border-radius: 10px;
        padding: 10px;
        display: inline-block;
        color: white;
    }}
    .custom-content {{
        margin-top: 20px;
        background: rgba(0, 0, 0, 0.6); /* 불투명 배경 추가 */
        border-radius: 10px;
        padding: 10px;
        display: inline-block;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def page():
    image_temp=None
    # 앱의 상태 관리
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"

    # 첫 번째 페이지: 동영상 업로드
    if st.session_state["page"] == "upload":
        # 스타일이 적용된 제목 추가
        st.markdown('<h1 class="custom-title">탑승자 행동분석을 통해 제공하는 차량 서비스</h1>', unsafe_allow_html=True)

        # 동영상 업로드 라벨에 불투명 배경 적용
        uploaded_video = st.file_uploader("\n", type=["mp4", "mov", "avi"])
        print(f"uploaded_video : {uploaded_video}")
        print(f"uploaded_video's type : {type(uploaded_video)}")
        
        if uploaded_video is not None: # 업로드된 파일 있다면
            temp_dir = tempfile.mkdtemp() # 동영상을 tempfile로 저장
            path = os.path.join(f"C:/Users/user/Documents/", uploaded_video.name)
            st.session_state['video_path']=path
            with open(path, "wb") as f:
                f.write(uploaded_video.read())  # 파일 데이터를 저장

            # 모델에 동영상 입력하여 행동 예측
            predicted_action,input_image = final_pipeline(path)
       
            print("input_image debugg ======",type(input_image))

            # 동영상 프레임 처리 및 저장
            frame_count, frame_list = video_to_frames(path, temp_dir)

            # 세션 상태에 결과 저장
            st.session_state["uploaded_video"] = path
            st.session_state["predicted_action"] = predicted_action
            st.session_state["input_image"] = input_image
            st.session_state['frame_list'] = frame_list
            st.session_state['image_temp'] = input_image
            

            # 성공 메시지를 불투명 배경으로 출력
            st.markdown('<div class="custom-success">동영상 업로드 성공!</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("다음 페이지로 이동"):
                st.session_state["page"] = "media"


    # 두 번째 페이지: 나머지 기능
    elif st.session_state["page"] == "media":
    # 분석 페이지 테마 설정
        st.markdown(
            """
            <style>
            .stApp {
                background: #1E1E1E; /* 어두운 회색 배경 */
                color: #FFFFFF; /* 흰색 텍스트 */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("Result")

        # 동영상 출력
        st.header("업로드한 동영상 재생")
        # st.write(st.session_state["uploaded_video"])
        play_video(st.session_state["uploaded_video"])

        # cv모델이 분류한 텍스트 출력
        st.header("slowfast 모델이 분류한 행동")
        st.text(st.session_state["predicted_action"])  # 모델이랑 연결...

        llm_api_key = os.getenv('LLM_API_KEY')
        play = Play_LLM(llm_api_key)

        model = 'gpt-4o'
        input_image=st.session_state['image_temp']
        predicted_action=st.session_state["predicted_action"]
        print("input image debuggg  ",type(st.session_state['image_temp']))
        original_width, original_height = input_image.size
    
    # 가로 길이 480에 맞춘 비율 계산
        new_width = 480
        scale_factor = new_width / original_width
        new_height = int(original_height * scale_factor)

        # 이미지 리사이즈
        resized_img = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 리사이즈된 이미지 저장

        image_path_=r".\temp_image.jpg"
        resized_img.save(image_path_, format="JPEG")
        

        first_text = play.get_image_response(
            model, action_=predicted_action
        )

        print(first_text)
        second_text = play.get_followup_response(model, first_text)
        print(second_text)

        tts_api_key = os.getenv('TTS_API_KEY')
        actor_id = '668f4f533ea5c6ce5e43fd48'

        play_tts(tts_api_key, actor_id, second_text)

        # LLM2TTS 오디오 재생
        st.header("LLM과 TTS를 거쳐 생성된 오디오 재생")
        audio_file = open('./final.wav',"rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

def video_to_frames(video_path, output_dir, prefix = "frame"):
    # 프레임을 저장할 경로 확인
    if not os.path.exists(output_dir): # 없으면 경로 생성
        os.makedirs(output_dir) 

    cap = cv2.VideoCapture(video_path) # 동영상 파일 열기
    if not cap.isOpened(): # 동영상 파일 열기 확인
        raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

    frame_count = 0 # 생성된 프레임 개수
    frame_list = [] # 저장된 프레임 경로 목록
    while True:
        ret, frame = cap.read()
        # ret: 읽기 성공 여부, frame: 읽은 프레임(이미지 데이터)
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # 각 프레임을 .jpg 파일로 저장
        frame_filename = os.path.join(output_dir, f"{prefix}_{frame_count:04d}.jpg")
        # 04d: 4자리 숫자로 된 프레임 번호(ex. 0001, 0002 ...)
        cv2.imwrite(frame_filename, frame)
        frame_list.append(frame_filename)
        frame_count += 1

    cap.release() # 동영상 파일에 대한 리소스 해제
    print(f"총 {frame_count}개의 프레임이 저장되었습니다.")
    return frame_count, frame_list

def play_video(video_path):
    output_memory_file = io.BytesIO() # 메모리에 파일처럼 데이터 저장할 수 있는 버퍼 생성
    output = av.open(output_memory_file, 'w', format="mp4") # 메모리 버퍼에
    stream = output.add_stream('h264', 30)
    stream.width = 224
    stream.height = 168
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17'}

    videos_count, video_files = video_to_frames(video_path, './back/')

    for i in range(videos_count):
        img = cv2.imread(video_files[i])  # Create OpenCV image for testing (resolution 192x108, pixel format BGR).
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')  # Convert image from NumPy Array to frame.
        packet = stream.encode(frame)  # Encode video frame
        output.mux(packet)

    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    output_memory_file.seek(0)
    st.video(output_memory_file)

if __name__ == '__main__':
    page()
