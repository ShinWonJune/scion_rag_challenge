# scion_rag_challenge

Basically to excute code, get inside of src and run it.
this use docker compose and devcontainer for ~~~
here is how to execute

0. open vscode and use dev container to set environments
1. Prepare the question docs and go to /workspace/search_science_on_chellenge
2. put text.csv under /workspace
3. execute below

```bash
TARGET_DOCUMENTS=50 python main.py batch test.csv  # 각 질문당 100개 문서
TARGET_DOCUMENTS=50 python main.py --use-vllm batch test.csv # vllm 사용시 
TARGET_DOCUMENTS=50 python main.py --use-vllm --use-pubmed batch test.csv # vllm 사용시 & pubmed에 검색할 때
```

4. you needs api key for gemini and science_on_api
5. take /workspace/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl to /workspace/data/expr
6. Move to src/ and build vector database for embeddings:

```bash
# Build vector database with specific GPU (recommended when vLLM is running)
python build_vectordb_search.py --gpu_id 2 # select gpu id 
python build_vectordb_search.py --config_path ../configs/query_encoder/config_bge_m3.json --gpu_id 2
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



python -m retrieval_system.main --config_json ../configs/query_encoder/config_gte-multilingual-base.json --questions_jsonl /workspace/data/expr singlehop_decompose.jsonl  --range  1-50 \ --top_k  50 \ --device  auto \ --schema_json  /workspace/configs/csv_schema/

python preprocess_and_generate_answer.py \
  --input_dir /workspace/results/retrival_docs/250908_235532 --max_rank 1 --parallel True
```
9. results are in /workspace/data/expr/final_result, once you execute preprocess_and_generate_answer.py, it will overwrite so make sure execute once or and the name of folder

```
python final_result.py

```


