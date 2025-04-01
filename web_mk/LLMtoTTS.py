import time
import requests
# from openai import OpenAI
import openai
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

class Play_LLM:
    def __init__(self, api_key):
        """
        클래스 초기화: OpenAI API 키를 설정합니다.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.first = ''
        self.second = ''

    def get_image_response(self, model,max_tokens=300,action_=None):
        """이미지 파일을 Base64로 인코딩"""
        image_path_=r".\temp_image.jpg"

        with open(image_path_, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("base64_image ======",type(base64_image))

        """이미지와 텍스트를 포함한 메시지로 모델 응답을 생성"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""사진 속 탑승자가 하고 있는 행동이 {action_}과 일치해?"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        self.first = response.choices[0].message.content

        return response.choices[0].message.content

    def get_followup_response(self, model, first_text, temperature=1):
        """
        이전 응답을 기반으로 추가적인 질문을 통해 권장사항을 생성합니다.
        """
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    'role': 'system',
                    'content': [
                        {
                            "type": "text",
                            "text": '너는 chat으로 답변하는 GPT가 아니라 자율주행 차량 탑승자에게 조언을 해주는 실제 조수야. 탑승자가 더 편리한 차내 서비스를 이용할 수 있도록 항상 말을 걸 준비가 되어 있고 설명 없이 실제 대사만을 응답해야 해.'
                        },
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "text",
                            "text": f'''누군가 사진을 보고 운전자의 상태를 다음과 같이 파악했습니다.: {first_text} . 
                            운전자가 무엇을 하고 있는지, 필요한 것은 무엇인지, 종합하여 한문장의 권유 대사를 작성해줘'''
                        },
                    ],
                },
            ],
        )
        followup_content = response.choices[0].message.content
        answer = str(followup_content).split("4")[-1]

        self.second = answer.split("refusal")[0]

        return answer.split("refusal")[0]

    def process_image_and_generate_advice(self, model, base64_image, max_tokens=300, temperature=1):
        """
        이미지와 텍스트 응답을 처리하고 운전자에게 권장 사항을 생성합니다.
        """
        try:
            # Step 1: Get the image response
            first_text = self.get_image_response(
                model=model, base64_image=base64_image, max_tokens=max_tokens)
            print(first_text)

            # Step 2: Get the follow-up response based on the first response
            second_text = self.get_followup_response(
                model=model, model_text=first_text, temperature=temperature)
            print(second_text)

            return second_text
        except Exception as e:
            return f"An error occurred: {e}"


def play_tts(tts_api_key, actor_id, llm_text):
    API_TOKEN = tts_api_key

    HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}
    
    '''
    if you want another voice, try to under code.
    # get my actor
    r = requests.get('https://typecast.ai/api/actor', headers=HEADERS)
    my_actors = r.json()['result']
    my_first_actor = my_actors[0]
    my_first_actor_id = my_first_actor['actor_id']
    '''
    
    # request speech synthesis
    r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
        'text': llm_text,
        'lang': 'auto',
        'actor_id': actor_id,
        'xapi_hd': True,
        'model_version': 'latest'
    })
    speak_url = r.json()['result']['speak_v2_url']

    # polling the speech synthesis result
    for _ in range(120):
        r = requests.get(speak_url, headers=HEADERS)
        ret = r.json()['result']
        # audio is ready
        if ret['status'] == 'done':
            # download audio file
            r = requests.get(ret['audio_download_url'])
            with open('final.wav', 'wb') as f:
                f.write(r.content)
            break
        else:
            print(f"status: {ret['status']}, waiting 1 second")
            time.sleep(1)


if __name__ == '__main__':
    # .env 파일 로드
    load_dotenv()
    # 환경 변수 읽기
    llm_api_key = os.getenv('LLM_API_KEY')

    # 이미지 파일 경로
    image_path = r"./image_reading_something.jpg"
    model = 'gpt-4o'

    play = Play_LLM(llm_api_key)

    first_text = play.get_image_response(model, image_path)
    print(first_text)
    second_text = play.get_followup_response(model, first_text)
    print(second_text)

    # llm_text = play.process_image_and_generate_advice(model, base64_image)

    # exit(0)

    tts_api_key = os.getenv('TTS_API_KEY')
    actor_id = '668f4f533ea5c6ce5e43fd48'
    play_tts(tts_api_key, actor_id, second_text)
