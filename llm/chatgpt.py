import json.decoder
import os
import openai
from utils.enums import LLM
import time


def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    # if model == LLM.TONG_YI_QIAN_WEN:
    #     import dashscope
    #     dashscope.api_key = OPENAI_API_KEY
    # else:
    #     openai.api_key = OPENAI_API_KEY
    #     openai.organization = OPENAI_GROUP_ID
    if OPENAI_API_KEY:
        # 직접 API 키 설정
        openai.api_key = OPENAI_API_KEY
        # 환경 변수로도 설정
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # organization 헤더는 기본적으로 설정하지 않음
    # API 키에 연결된 기본 조직이 자동으로 사용됨
    openai.organization = None
    
    # API 키가 설정되었는지 확인
    if not openai.api_key and "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    
    # API 키가 여전히 없으면 경고 출력
    if not openai.api_key:
        print("WARNING: OpenAI API 키가 설정되지 않았습니다. API 호출이 실패할 수 있습니다.")
        print("API 키를 환경 변수(OPENAI_API_KEY)로 설정하거나 명령줄 인수로 전달하세요.")


def ask_completion(model, batch, temperature):
    response = openai.Completion.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def ask_chat(model, batch, temperature=0, n=1):
    # 문자열을 올바른 메시지 형식으로 변환
    if isinstance(batch, str) or (isinstance(batch, object) and hasattr(batch, 'shape') and len(batch.shape) == 0):
        # 단일 문자열인 경우
        messages = [{"role": "user", "content": str(batch)}]
    elif isinstance(batch, list):
        if len(batch) > 0 and isinstance(batch[0], str):
            # 문자열 리스트인 경우
            messages = [{"role": "user", "content": str(item)} for item in batch]
        else:
            # 이미 적절한 형식일 경우
            messages = batch
    else:
        # 기타 형식은 문자열로 변환
        messages = [{"role": "user", "content": str(batch)}]
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n
        )
        
        if n == 1:
            response_clean = [choice["message"]["content"] for choice in response["choices"]]
        else:
            response_clean = [[choice["message"]["content"] for choice in response["choices"]]]
        
        return dict(
            response=response_clean,
            total_tokens=response["usage"]["total_tokens"]
        )
    except Exception as e:
        print(f"Chat API 호출 오류: {e}")
        print(f"전달된 메시지: {messages}")
        return None


def ask_llm(model, batch, temperature=0, n=1):
    try:
        if model in [LLM.TEXT_DAVINCI_003]:
            return ask_completion(model, batch, temperature)
        else:  # gpt-3.5-turbo, gpt-4
            return ask_chat(model, batch, temperature, n)
    except openai.error.RateLimitError:
        print(f"Rate limit exceeded. Sleeping for 10 seconds.")
        time.sleep(10)
        return ask_llm(model, batch, temperature, n)
    except Exception as e:
        print(f"Error: {e}")
        return None

