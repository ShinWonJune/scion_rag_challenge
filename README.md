# scion_rag_challenge

Basically to excute code, get inside of src and run it.
this use docker compose and devcontainer for ~~~
here is how to execute

0. open vscode and use dev container to set environments
1. Prepare the question docs and go to /workspace/search_science_on_chellenge
2. put text.csv under /workspace
3. execute below

```bash
# api search
TARGET_DOCUMENTS=50 python main.py batch test.csv  # 각 질문당 100개 문서
TARGET_DOCUMENTS=50 python main.py --use-vllm batch test.csv # vllm 사용시 
TARGET_DOCUMENTS=50 python main.py --use-vllm --use-pubmed batch test.csv # vllm 사용시 & pubmed에 검색할 때
TARGET_DOCUMENTS=20 python main.py --input-dir /app/data/miracl/questions --keyword-lang english --use-wiki batch subquestion.csv 
# --input-dir 은 quetstion (subquestion.csv) 의 경로를 지정
# --use-wiki: 위키피디아 사용
# --keyword-lang LANG: 키워드추출 언어 지정 (all, english, korean)
# 키워드 추출 언어를 제한한 이유:
# wikipedia의 경우, 한국 위키피디아 주소와 영어 위키피디아 주소가 구분됨. 검색어의 언어에 따라서 api 주소도 변경해야함. 
# 그러나 현재 wikipedia_api_client.py (위키 검색 모듈) 은 동적인 주소 변환을 지원하지 않음. 현재 영어 wiki 로 설정된 상태.
# 따라서 --keyword-lang 의 arg를 추가하여 키워드 생성 언어를 지정해줌. 영어 wiki에는 영어 질문만 할 수 있도록.

```

4. you needs api key for gemini and science_on_api
5. take /workspace/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl to /workspace/data/expr
6. Move to src/ and build vector database for embeddings:

```bash
# Build vector database with specific GPU (recommended when vLLM is running)
# specifiy searched documents directory with  --docs_jsonl_path 
python build_vectordb_search.py --gpu_id 2 # select gpu id 
python build_vectordb_search.py --config_path ../configs/query_encoder/config_bge_m3.json  --docs_jsonl_path ../search_science_on_challenge/outputs/search_documents_20250921_111520.jsonl


```

7. Execute line by line in main.ipynb
8. if you want to execute fast you and you the command in the ipynb shell to bash directly ipynb require to setup to use gpu

> Other Option

```
python multi_hop_to_single_hop.py \
  --input /workspace/data/rag_test_data/questions.jsonl \
  --output /workspace/data/expr/singlehop_decompose.jsonl \
  --question_field question --context_field context \
  --mode decompose --model gemini-2.5-flash


CUDA_VISIBLE_DEVICES=2 #when using vLLM
python -m retrieval_system.main --config_json ../configs/query_encoder/config_gte-multilingual-base.json --questions_jsonl /workspace/data/expr singlehop_decompose.jsonl --vectordb_csv ../results/vectordb/250921_112956/vector_db_bge_m3_test_BAAI_bge-m3.csv \ --range  1-50 \ --top_k  50 \ --device  auto \ --schema_json  /workspace/configs/csv_schema/

python preprocess_and_generate_answer.py \
  --input_dir /workspace/results/retrival_docs/250908_235532 --max_rank 1 --parallel True
# output in ../results/final_answers/

```
9. After runnigng preprocess_and_generate_answer.py, results are in /workspace/data/expr/final_result, once you execute preprocess_and_generate_answer.py, it will overwrite so make sure execute once or and the name of folder.

10. Gathering answers as a csv file with final_result.py (scienceon) or final_result_pubmed.py (pubmed)

```
python final_result.py  #output in ../results/competition_submission

python final_result_pubmed.py  #output in ../results/pumbedqa_final_answers
```


11. Evaluation with BLEU, METEOR

```
python evaluate_pubmedqa.py --input_path /app/results/pubmedqa_final_answers/pubmedqa_final_predictions_250921_113523.csv
```