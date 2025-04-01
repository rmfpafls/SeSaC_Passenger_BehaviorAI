from flask import Flask, request, render_template, jsonify
from LLMtoTTS import Play_LLM, play_ttx
import os
from dotenv import load_dotenv

# Flask 앱 생성
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 영상 파일 받기
    if 'video' not in request.files:
        return jsonify({"error": "No video file found."}), 400
    video_file = request.files['video']
    video_path = os.path.join('static', 'uploaded_video.mp4')
    video_file.save(video_path)

    # 비디오 파일에서 프레임 추출 후 인코딩
    base64_frame = llm_play.extract_and_encode_frame(video_path)

    # LLM을 이용해 이미지 분석 및 텍스트 생성
    model = 'gpt-4o'
    try:
        first_text = llm_play.get_image_response(model, base64_frame)
        second_text = llm_play.get_followup_response(model, first_text)
    except Exception as e:
        return jsonify({"error": f"Error with LLM processing: {e}"}), 500

    # TTS로 변환
    try:
        play_ttx(second_text)
    except Exception as e:
        return jsonify({"error": f"Error with TTS processing: {e}"}), 500

    # TTS 음성 파일 URL 반환
    return jsonify({"message": second_text, "audio_url": "/static/final.wav"})


if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()

    # 환경 변수 읽기
    llm_api_key = os.getenv('LLM_API_KEY')
    tts_api_key = os.getenv('TTS_API_KEY')
    actor_id = '668f4f533ea5c6ce5e43fd48'

    # LLM 클래스 초기화
    llm_play = Play_LLM(llm_api_key)
    app.run(debug=True)
