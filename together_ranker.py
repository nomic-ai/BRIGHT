import os
import re
import copy
import time
import logging
import math # Added for ceil
from typing import List, Dict, Any, Tuple
from functools import wraps
from tqdm import tqdm # Added import
from openai import OpenAI

import together

logger = logging.getLogger(__name__)


def call_api(func):
    """Decorator to handle API call retries."""
    # The decorator needs to preserve the signature of the wrapped function
    @wraps(func)
    def wrapper(*args, **kwargs):
        count = 0
        while True:
            try:
                count += 1
                # Call the wrapped function with its original arguments
                output = func(*args, **kwargs)
                break
            except Exception as e:
                logger.warning(f"Exception calling Together API: {e}")
                if "rate limit" in str(e).lower():
                    logger.info("Rate limit exceeded, waiting 10 secs...")
                    time.sleep(10)
                elif count < 5:
                    logger.info("Retrying API call...")
                    time.sleep(5)
                else:
                    logger.error("API call failed after 5 retries.")
                    output = None
                    break
        return output
    return wrapper

TASK2PROMPT = {
    "biology": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "earth_science": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "economics": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "psychology": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "robotics": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "stackoverflow": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "sustainable_living": "A document is relevant if it contains information that helps answer or address the query.  A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n",
    "aops": "We want to find different but similar math problems to the query shared above. A document is relevant if it uses the same class of functions and shares **any** overlapping techniques.\n",
    "leetcode": "I am looking to find different problems that share similar data structures (of any kind) or algorithms (e.g. DFS, DP, sorting, traversals, etc.) to the query above. I am looking for problems that share one or both of these similarities. A document is relevant if there was a textbook on leetcode problems, this would be in the same book even though it could be in a different chapter.\n",
    "pony": "I will use the programming language pony. But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match any part (even basic parts) of the code I will have to write for the program.\n",
    "theoremqa_questions": "We want to find a document which uses the same mathematical process as the query above. A document is relevant if it uses the same mathematical process as the query.\n",
    "theoremqa_theorems": "We want to find a document which uses the same mathematical process as the query above. A document is relevant if it uses the same mathematical process as the query.\n",
}

class TogetherListwiseReranker:
    """Reranks documents using a sliding window approach with the Together AI API."""

    SYSTEM_PROMPT = """You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."""

    def __init__(self, model_name: str, task: str, window_size: int = 10, stride: int = 5, together_api: bool = False, api_key="EMPTY", base_url="http://localhost:8000/v1"):
        """
        Initializes the reranker.

        Args:
            model_name: The name of the model to use via Together AI.
            window_size: The number of documents in each reranking window.
            stride: The step size for the sliding window.
        """
        if not together_api:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            if not os.environ.get("TOGETHER_API_KEY"):
                raise ValueError("TOGETHER_API_KEY environment variable not set.")
            together.api_key = os.environ["TOGETHER_API_KEY"]
            self.client = together.Together()
        self.task = task
        self.together_api = together_api
        self.model_name = model_name
        self.window_size = window_size
        self.stride = stride
        # Basic check for compatibility, might need adjustment based on specific models
        self.max_tokens = 4096 
        # Adjust output tokens based on expected format '[1] > [2] > ...'
        # It depends on window size, roughly estimate needed tokens
        self.num_output_tokens = window_size * 6 # Estimate: ~6 chars per '[N] > '
        logger.info(f"Initialized TogetherListwiseReranker with model: {model_name}, window: {window_size}, stride: {stride}")

    def _format_prompt(self, query: str, docs_in_window: List[Tuple[str, str]]) -> str:
        """Formats the user prompt for the LLM based on the new structure."""
        doc_string = ""
        # The user provided doc_ids, but our input is (doc_id, doc_text).
        # We will use 1-based index for the prompt as in the example.
        for idx, (_, doc_text) in enumerate(docs_in_window):
            # Use 1-based index for the prompt passages
            # Calculate cleaned text first to avoid f-string syntax error
            cleaned_text = re.sub(r'\n+', ' ', doc_text)
            doc_string += f"[{idx + 1}] {cleaned_text}\n"
        
        # Construct the user prompt based on the provided example
        prompt = (
            f"I will provide you with {len(docs_in_window)} passages,"
            f"each indicated by a numerical identifier [].\n"
            f"Rank the passages based on their relevance to the query: {query}.\n"
            f"{TASK2PROMPT[self.task]}"
            f"{doc_string}\n"
            f"Search Query: {query}\n"
            f"Rank the {len(docs_in_window)} passages above based on their "
            f"relevance to the query. All the"
            f"passages should be included and listed "
            f"using identifiers, in descending order of "
            f"relevance. The output format should be [] > [], "
            f"e.g., [4] > [2]. Think before responding only with the ranking "
            f"results, do not say any word or explain."
        )
        return prompt

    def _clean_response(self, response: str) -> str:
        """Extracts the permutation string (e.g., '[1] > [3] > [2]') from the response,
           assuming it follows the last '</think>' tag."""
        # Find the position of the last '</think>' tag
        last_think_tag_pos = response.rfind('</think>')
        
        if last_think_tag_pos != -1:
            # If the tag is found, extract the substring after it
            cleaned = response[last_think_tag_pos + len('</think>'):].strip()
        else:
            # If the tag is not found, assume the entire response might be the ranking
            # or handle cases where the model didn't follow the expected format.
            logger.warning("Could not find '</think>' tag in response. Attempting to parse the full response.")
            cleaned = response.strip()
            
        # Optional: Add more cleaning steps if needed after extraction
        # e.g., removing potential leading non-ranking characters if the model is inconsistent
        return cleaned

    def _remove_duplicate(self, response_list: List[int]) -> List[int]:
        """Removes duplicate indices from the permutation list."""
        new_response = []
        seen = set()
        for c in response_list:
            if c not in seen:
                new_response.append(c)
                seen.add(c)
        return new_response

    def _parse_permutation(self, permutation_str: str, window_len: int) -> List[int]:
        """Parses the '[id] > [id] > ...' permutation string into a list of 0-based indices."""
        try:
            # Extract numbers from the brackets
            ids = re.findall(r'\[(\d+)\]', permutation_str)
            if not ids:
                 logger.warning(f"Could not find bracketed IDs in permutation string: '{permutation_str}'. Falling back to original order.")
                 return list(range(window_len))
                 
            # Convert to 1-based integer indices
            response = [int(id_str) - 1 for id_str in ids]
            
            response = self._remove_duplicate(response)
            
            # Validate indices are within the expected range (0 to window_len - 1)
            valid_response = [idx for idx in response if 0 <= idx < window_len]
            
            # Add missing indices to the end in their original order
            original_indices = list(range(window_len))
            missing_indices = [idx for idx in original_indices if idx not in valid_response]
            final_permutation = valid_response + missing_indices
            
            if len(final_permutation) != window_len:
                 logger.warning(f"Permutation length mismatch after parsing '{permutation_str}'. Expected {window_len}, got {len(final_permutation)}. Falling back.")
                 return list(range(window_len))

            return final_permutation
        except Exception as e:
            logger.error(f"Error parsing permutation string '{permutation_str}': {e}. Falling back to original order.")
            return list(range(window_len))

    @call_api
    def _call_together_api(self, user_prompt: str) -> str | None:
        """Calls the Together AI chat completion endpoint."""
        if not self.together_api:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    # extra_body={
                    #     "chat_template_kwargs": {"enable_thinking": False}
                    # },
                    # temperature=0.0, # Deterministic for ranking
                    # top_p=1.0,
                    # repetition_penalty=1.0,
                    # Adjust stop sequences if needed for the chat format
                    # stop=["\n\n"] 
                )
                if response.choices:
                    # Access message content correctly for chat completions
                    return response.choices[0].message.content
                else:
                    logger.warning("Received empty choices from Together API chat completions.")
                    return None
            except Exception as e:
                # Catch potential API errors specific to chat completions if they differ
                logger.error(f"Error calling Together chat completions API: {e}")
                # Let the @call_api decorator handle retries/failure
                raise e
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    # Adjust stop sequences if needed for the chat format
                    # stop=["\n\n"] 
                )
                if response.choices:
                    # Access message content correctly for chat completions
                    return response.choices[0].message.content
                else:
                    logger.warning("Received empty choices from Together API chat completions.")
                    return None
            except Exception as e:
                # Catch potential API errors specific to chat completions if they differ
                logger.error(f"Error calling Together chat completions API: {e}")
                # Let the @call_api decorator handle retries/failure
                raise e

    def rerank(self, docs: List[Tuple[str, str]], query: str, topk: int) -> List[str]:
        """
        Reranks the documents using the sliding window approach.

        Args:
            docs: A list of tuples, where each tuple is (doc_id, doc_text).
            query: The search query string.
            topk: The number of top documents to return.

        Returns:
            A list of document IDs reranked according to relevance, up to topk.
        """
        if not docs:
            return []

        rerank_candidates = copy.deepcopy(docs)
        rank_start = 0
        rank_end = len(rerank_candidates)
        window_size = min(self.window_size, rank_end)
        stride = self.stride

        if stride <= 0:
             logger.warning("Stride must be positive. Falling back to no reranking.")
             return [doc_id for doc_id, _ in rerank_candidates[:topk]]

        end_pos = rank_end
        start_pos = max(rank_start, rank_end - window_size)

        logger.info(f"Starting sliding window rerank for query: '{query[:50]}...' with {len(docs)} docs.")

        # Calculate total iterations for tqdm
        total_iterations = math.ceil((rank_end - rank_start) / stride)

        # Initialize tqdm progress bar
        progress_bar = tqdm(total=total_iterations, desc="Reranking windows", unit="window")

        iteration = 0
        while end_pos > rank_start:
            iteration += 1
            current_start_pos = max(rank_start, start_pos)
            window_docs = rerank_candidates[current_start_pos:end_pos]
            window_len = len(window_docs)

            if window_len == 0:
                logger.warning("Empty window encountered, breaking loop.")
                break

            logger.debug(f"Iteration {iteration}: Window [{current_start_pos}, {end_pos}), Size: {window_len}")

            # Create user prompt for the current window
            user_prompt = self._format_prompt(query, window_docs)

            # Call LLM using chat completion endpoint
            permutation_str_raw = self._call_together_api(user_prompt)

            if permutation_str_raw is None:
                logger.warning(f"API call failed for window [{current_start_pos}, {end_pos}). Skipping permutation.")
                # Skip permutation application if API fails
            else:
                logger.debug(f"Raw LLM Output: {permutation_str_raw}")
                cleaned_permutation_str = self._clean_response(permutation_str_raw)
                logger.debug(f"Cleaned LLM Output: {cleaned_permutation_str}")
                permutation = self._parse_permutation(cleaned_permutation_str, window_len)
                logger.debug(f"Parsed Permutation (0-based): {permutation}")

                # Reorder documents within the main list based on the permutation
                original_window_copy = copy.deepcopy(window_docs)
                for new_idx_in_window, old_idx_in_window in enumerate(permutation):
                     if 0 <= old_idx_in_window < len(original_window_copy):
                        rerank_candidates[current_start_pos + new_idx_in_window] = original_window_copy[old_idx_in_window]
                     else:
                         logger.error(f"Invalid index {old_idx_in_window} in permutation for window size {len(original_window_copy)}")

            # Slide the window
            end_pos -= stride
            start_pos -= stride

            # Update progress bar
            progress_bar.update(1)

        # Ensure the progress bar closes cleanly
        progress_bar.close()

        logger.info("Sliding window rerank finished.")
        # Return only the topk document IDs
        return [doc_id for doc_id, _ in rerank_candidates[:topk]]

# Example usage (optional, for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Ensure TOGETHER_API_KEY is set as an environment variable
    # export TOGETHER_API_KEY='your_api_key'
    
    # Example documents (doc_id, doc_text)
    docs_to_rerank = [
        ("doc1", "The quick brown fox jumps over the lazy dog."),
        ("doc2", "A lazy dog sits under the tree."),
        ("doc3", "Foxes are omnivorous mammals belonging to several genera of the family Canidae."),
        ("doc4", "The study of dogs is known as cynology."),
        ("doc5", "Quick feet help the fox escape predators."),
        ("doc6", "Brown bears are found in North America."),
        ("doc7", "Lazy rivers flow slowly."),
        ("doc8", "Jumping requires strong leg muscles."),
        ("doc9", "The quicksand trapped the unlucky traveler."),
        ("doc10", "Dog breeds vary greatly in size and temperament."),
        # ("doc11", "A fast brown animal lept."),
        # ("doc12", "Sloths are known for being lazy."),
    ]
    query_str = "information about quick brown foxes and lazy dogs"
    
    try:
        # Use a relatively cheap/fast model for testing
        # Find suitable models via `together models list`
        ranker = TogetherListwiseReranker(model_name="gpt-4.1", 
                                          window_size=4, 
                                          stride=2, 
                                          task="biology",
                                          api_key="",
                                          base_url=None)
        reranked_ids = ranker.rerank(docs=docs_to_rerank, query=query_str, topk=10)
        print("\nReranked Document IDs:", reranked_ids)
        
        # Print reranked docs for verification
        reranked_docs_map = {d[0]: d[1] for d in docs_to_rerank}
        print("\nReranked Documents:")
        for i, doc_id in enumerate(reranked_ids):
            print(f"{i+1}. [{doc_id}] {reranked_docs_map[doc_id]}")
            
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
