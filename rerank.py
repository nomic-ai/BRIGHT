import copy
import os
import re
import time
import json
from tqdm import tqdm
import argparse
from datasets import load_dataset
import torch
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai_ranker import OpenAIReranker
from retrievers import calculate_retrieval_metrics
from together_ranker import TogetherListwiseReranker

import functools
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    return contents.replace(f"{lan}\n", '')


def call_api(func):
    count = 0
    while True:
        try:
            count += 1
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                logger.info("Rate limit exceeded, waiting 10 secs and retrying...")
                time.sleep(10)
            elif count < 5:
                logger.info("Encountered error, retrying...")
                time.sleep(5)
            else:
                logger.info("Skipping generation due to unknown error after 5 retries.")
                output = None
                break
    return output


def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    if include_system:
        chat = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages:\n'
                  f'```json\n'
                  f'[...]\n'
                  f'```')
        func = functools.partial(
            self.model.messages.create,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.version,
            temperature=0.8,
            top_p=0.8
        )
        message = call_api(func)
        response = json.loads(message.model_dump_json(indent=2))
        ranks = extract_program(response['content'][0]['text'],lan='json')
        return [indices_map[r] for r in ranks]


class OpenAIModel:
    def __init__(self, model_name, temperature=0.8, top_p=0.8):
        import openai
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = 2048

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages, sorted from the most to least relevant passage. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format.'
                  )
        inputs = format_chat(prompt, system_message="You are a helpful assistant")
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        def parse_json(text):
            matches = re.findall(r"(?:```json\s*)(.+)(?:```)", text, re.DOTALL)
            if len(matches) > 0:
                try:
                    return json.loads(matches[-1].strip())
                except:
                    return None
            return None

        output = call_api(func)
        if output is not None:
            response = parse_json(output.choices[0].message.content)
            if response is None:
                return None
            return [indices_map[r] for r in response if r in indices_map]
            # return output.choices[0].message.content
        return None


class STReranker:
    def __init__(self, model_name, batch_size=8):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        inputs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(inputs, batch_size=self.batch_size)
        ranking = {doc["id"]: score.item() for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking


def rerank_single_query(qid, scores, model, documents, examples, args):
    logger.debug(f"Reranking qid: {qid}")
    try:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.input_k]
        
        if isinstance(model, TogetherListwiseReranker):
            # Together ranker expects List[Tuple[str, str]]
            docs_for_rerank = [(did, documents[did]) for did, _ in sorted_scores]
        elif isinstance(model, (ClaudeModel, OpenAIModel)):
            # Other models might expect List[List[str, str]] - adjust if needed
            docs_for_rerank = [[did, documents[did]] for did, _ in sorted_scores]
        else:
            # Default or handle other model types if necessary
            docs_for_rerank = [(did, documents[did]) for did, _ in sorted_scores]

        reranked_ids = model.rerank(docs=docs_for_rerank, query=examples[qid]['query'], topk=args.k)
        
        # Assign descending scores based on the new order
        final_scores = {doc_id: args.k - i for i, doc_id in enumerate(reranked_ids)}
        logger.debug(f"Finished reranking qid: {qid}")
        return qid, final_scores
    except Exception as e:
        logger.error(f"Error reranking qid {qid}: {e}", exc_info=True)
        # Return qid and empty scores on error to avoid blocking others
        return qid, {}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--llm', type=str, default=None, help="Model name for Claude, OpenAI, or Sentence Transformer rerankers (e.g., 'claude-3-opus-20240229', 'gpt-4-turbo', 'mixedbread-ai/mxbai-rerank-xsmall-v1')")
    parser.add_argument('--together_model', type=str, default=None, help="Model name for TogetherListwiseReranker (e.g., 'mistralai/Mixtral-8x7B-Instruct-v0.1')")
    parser.add_argument('--openai', action="store_true")
    parser.add_argument('--window_size', type=int, default=10, help="Window size for TogetherListwiseReranker sliding window.")
    parser.add_argument('--stride', type=int, default=5, help="Stride for TogetherListwiseReranker sliding window.")
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--rerank_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--together_api', action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(f"Running reranking for {args.task=}")

    if os.path.exists(args.rerank_score_file):
        print(f"Rerank score file {args.rerank_score_file} already exists.")
        exit()

    raw_examples = load_dataset('xlangai/bright', 'examples')[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents')[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents')[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']
    with open(args.score_file) as f:
        all_scores = json.load(f)
    new_scores = copy.deepcopy(all_scores)

    model = None
    if args.together_model:
        logger.info(f"Using TogetherListwiseReranker with model: {args.together_model}")
        if args.openai:
            try:
                model = OpenAIReranker(model_name=args.together_model, 
                                       task=args.task,
                                       window_size=args.window_size, 
                                       stride=args.stride, 
                                       api_key=os.getenv("OPENAI_API_KEY"),
                                       base_url=None,
                                       together_api=args.together_api
                                    )
            except ValueError as e:
                logger.error(f"Error initializing OpenAIReranker: {e}")
                exit(1)
        else:
            try:
                model = TogetherListwiseReranker(model_name=args.together_model, 
                                                task=args.task,
                                                window_size=args.window_size, 
                                                stride=args.stride, 
                                                together_api=args.together_api
                                                )
            except ValueError as e:
                logger.error(f"Error initializing TogetherListwiseReranker: {e}")
                exit(1)
    elif args.llm:
        if 'claude' in args.llm:
            logger.info(f"Using ClaudeModel with version: {args.llm}")
            model = ClaudeModel(version=args.llm)
        elif "gpt" in args.llm:
            logger.info(f"Using OpenAIModel with model name: {args.llm}")
            model = OpenAIModel(model_name=args.llm)
        else:
            logger.info(f"Using STReranker with model name: {args.llm}")
            model = STReranker(model_name=args.llm)
    else:
        logger.error("No reranker specified. Please provide --llm or --together_model.")
        exit(1)

    reranked_scores = {}
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        logger.info(f"Submitting {len(all_scores)} queries for parallel reranking...")
        for qid, scores in all_scores.items():
            futures.append(executor.submit(rerank_single_query, qid, scores, model, documents, examples, args))
        
        logger.info(f"Waiting for reranking tasks to complete...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reranking Queries"):
            try:
                qid_result, final_scores_result = future.result()
                if final_scores_result: 
                    reranked_scores[qid_result] = final_scores_result
            except Exception as e:
                logger.error(f"Error retrieving result from future: {e}", exc_info=True)

    os.makedirs(os.path.dirname(args.rerank_score_file), exist_ok=True)
    with open(args.rerank_score_file, 'w') as f:
        json.dump(reranked_scores, f, indent=2)

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for i in e["excluded_ids"]:
            if i in documents:
                ground_truth[e['id']][i] = 0

    results = calculate_retrieval_metrics(results=reranked_scores, qrels=ground_truth)
    with open(args.rerank_score_file.replace(".json", "_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
