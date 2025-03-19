import argparse
import os
import json

import openai
from tqdm import tqdm

from llm.chatgpt import init_chatgpt, ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader

from utils.post_process import process_duplication, get_sqls

QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default=None)  # 기본값을 None으로 변경
    parser.add_argument("--model", type=str, choices=[LLM.TEXT_DAVINCI_003, 
                                                      LLM.GPT_35_TURBO,
                                                      LLM.GPT_35_TURBO_0613,
                                                      # LLM.TONG_YI_QIAN_WEN,
                                                      LLM.GPT_35_TURBO_16K,
                                                      LLM.GPT_4],
                        default=LLM.GPT_35_TURBO)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mini_index_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set")
    parser.add_argument("--db_dir", type=str, default="dataset/spider/database")
    args = parser.parse_args()

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"
        
    # API 키 설정 부분
    # 1. 명령줄 인수에서 API 키 가져오기
    api_key = args.openai_api_key
    
    # 2. 없으면 환경 변수에서 가져오기
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    # 3. API 키가 있는지 확인
    if not api_key:
        print("OpenAI API 키가 제공되지 않았습니다.")
        print("API 키를 명령줄 인수(--openai_api_key)로 전달하거나")
        print("환경 변수(OPENAI_API_KEY)로 설정하세요.")
        exit(1)
    
    # API 키 설정
    openai.api_key = api_key
    
    # 명시적으로 환경 변수에도 설정 (다른 모듈에서 필요할 수 있음)
    os.environ["OPENAI_API_KEY"] = api_key
    
    # organization 설정 제거 (API 키에 연결된 기본 조직 사용)
    openai.organization = None
    
    print(f"API 키 설정 완료: {api_key[:5]}...{api_key[-5:]}")
    
    # 데이터 로드
    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    # init openai api - organization은 사용하지 않음
    init_chatgpt(api_key, None, args.model)

    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    if args.mini_index_path:
        mini_index = json.load(open(args.mini_index_path, 'r'))
        questions = [questions[i] for i in mini_index]
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    token_cnt = 0
    with open(out_file, mode) as f:
        for i, batch in enumerate(tqdm(question_loader)):
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            
            # 배치 형식 변환 (문자열 → 메시지 객체)
            formatted_batch = batch
            if args.batch_size == 1 and isinstance(batch, str):
                # 단일 문자열을 메시지 형식으로 변환
                formatted_batch = [{"role": "user", "content": batch}]
            elif isinstance(batch[0], str):
                # 문자열 리스트를 메시지 형식으로 변환
                formatted_batch = [{"role": "user", "content": item} for item in batch]
            
            try:
                res = ask_llm(args.model, formatted_batch, args.temperature, args.n)
                # None 반환 처리
                if res is not None:
                    token_cnt += res.get("total_tokens", 0)  # 안전하게 접근
                    if args.n == 1:
                        for sql in res["response"]:
                            # remove \n and extra spaces
                            sql = " ".join(sql.replace("\n", " ").split())
                            sql = process_duplication(sql)
                            # python version should >= 3.8
                            if sql.startswith("SELECT"):
                                f.write(sql + "\n")
                            elif sql.startswith(" "):
                                f.write("SELECT" + sql + "\n")
                            else:
                                f.write("SELECT " + sql + "\n")
                    else:
                        results = []
                        cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                        for sqls, db_id in zip(res["response"], cur_db_ids):
                            processed_sqls = []
                            for sql in sqls:
                                sql = " ".join(sql.replace("\n", " ").split())
                                sql = process_duplication(sql)
                                if sql.startswith("SELECT"):
                                    pass
                                elif sql.startswith(" "):
                                    sql = "SELECT" + sql
                                else:
                                    sql = "SELECT " + sql
                                processed_sqls.append(sql)
                            result = {
                                'db_id': db_id,
                                'p_sqls': processed_sqls
                            }
                            final_sqls = get_sqls([result], args.n, args.db_dir)

                            for sql in final_sqls:
                                f.write(sql + "\n")
                else:
                    print(f"Warning: API 호출 실패 (인덱스 {i})")
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

