from typing import List, Tuple
from together_ranker import TogetherListwiseReranker, TASK2PROMPT
import logging
import os

class OpenAIReranker(TogetherListwiseReranker):
    """Reranker using OpenAI's models, with a prompt optimized for chain-of-thought reasoning."""

    def _format_prompt(self, query: str, docs_in_window: List[Tuple[str, str]]) -> str:
        """Formats the user prompt for the LLM with chain-of-thought instructions."""
        passages_str = ""
        for idx, (doc_id, doc_text) in enumerate(docs_in_window):
            # Using 1-based indexing for passages as it's common in prompts
            # Not stripping newlines from doc_text to preserve original formatting for the model
            passages_str += f"[{idx + 1}] (ID: {doc_id}) {doc_text}\n\n"

        # Ensure there's no trailing newline if passages_str is empty, though unlikely.
        passages_str = passages_str.strip()
        if "gpt-4.1" in self.model_name:
            prompt = f"""
            Given a query and a list of passages, your task is to re-rank these passages based on their relevance to the query. {TASK2PROMPT[self.task]} 

            Please perform the following steps:
            1. **Understand the Query**: First, carefully read and understand the user's query to identify the core information need.
            2. **Analyze Each Passage**: For each passage, critically evaluate its content and determine how well it addresses the query. Consider factors like:
                - Directness of the answer
                - Completeness of the information
                - Presence of supporting evidence or details
                - Absence of irrelevant or distracting information
            3. **Compare and Contrast**: Compare the passages against each other. Identify which passages are more relevant and why. Note any subtle differences in relevance.
            4. **Reasoning for Ranking**: Explicitly state your reasoning for the rank you assign to each passage. Explain why a passage is ranked higher or lower than others. This step-by-step thought process is crucial.
            5. **Assign Ranks**: Based on your analysis and reasoning, assign a unique rank to each passage, starting from 1 for the most relevant. 

            **Output Format:**
            Your final output must be a list of ranks, corresponding to the original order of the passages. For example, if there are 3 passages, and you decide the second passage is most relevant, the first is second most relevant, and the third is least relevant, your output should be:
            [2] > [1] > [3]

            No other text or explanation should be present in the final output, only the list of ranks.

            **Query:**
            {query}

            **Passages:**
            {passages_str}

            **Your Step-by-Step Reasoning (before the final output list):**
            [Provide your detailed thought process here, analyzing each passage and justifying your ranking decisions. This section will not be part of the final output but helps in arriving at the correct ranking.]

            **Ranks (only this list will be parsed):**
            """
        else:
            prompt = f"""
            Given a query and a list of passages, your task is to re-rank these passages based on their relevance to the query. {TASK2PROMPT[self.task]} 

            Please perform the following steps:
            1. **Understand the Query**: First, carefully read and understand the user's query to identify the core information need.
            2. **Analyze Each Passage**: For each passage, critically evaluate its content and determine how well it addresses the query. Consider factors like:
                - Directness of the answer
                - Completeness of the information
                - Presence of supporting evidence or details
                - Absence of irrelevant or distracting information
            3. **Compare and Contrast**: Compare the passages against each other. Identify which passages are more relevant and why. Note any subtle differences in relevance.
            4. **Assign Ranks**: Based on your analysis and reasoning, assign a unique rank to each passage, starting from 1 for the most relevant. 

            **Output Format:**
            Your final output must be a list of ranks, corresponding to the original order of the passages. For example, if there are 3 passages, and you decide the second passage is most relevant, the first is second most relevant, and the third is least relevant, your output should be:
            [2] > [1] > [3]

            No other text or explanation should be present in the final output, only the list of ranks.

            **Query:**
            {query}

            **Passages:**
            {passages_str}
            """
        return prompt

# Example usage (optional, for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Changed to INFO for less verbose default
    # Ensure OPENAI_API_KEY is set as an environment variable for testing
    # export OPENAI_API_KEY='your_openai_api_key'
    
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
    ]
    query_str = "information about quick brown foxes and lazy dogs"
    
    try:
        # Retrieve API key from environment variable
        # The TogetherListwiseReranker base class should handle the api_key and api_base arguments
        # if they are passed to its __init__ method.
        # Since OpenAIReranker now uses parent's __init__, we can pass these.
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger = logging.getLogger(__name__)
            logger.warning("OPENAI_API_KEY environment variable not set. API calls might fail.")
            # Or raise ValueError("OPENAI_API_KEY must be set for testing")

        # Instantiate OpenAIReranker
        # model_name defaults to "gpt-4-turbo-preview" in parent if not specified.
        # The parent __init__ takes api_key and api_base (which is called base_url there)
        ranker = OpenAIReranker( 
            model_name="o3-mini",
            window_size=4, 
            stride=2,
            api_key=openai_api_key,
            base_url=None,
            task="biology" # Provide a task value as parent expects it
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Testing OpenAIReranker with model: {ranker.model_name}")
        reranked_ids = ranker.rerank(docs=docs_to_rerank, query=query_str, topk=10)
        print("\nReranked Document IDs:", reranked_ids)
        
        # Print reranked docs for verification
        reranked_docs_map = {d[0]: d[1] for d in docs_to_rerank}
        print("\nReranked Documents:")
        for i, doc_id in enumerate(reranked_ids):
            print(f"{i+1}. [{doc_id}] {reranked_docs_map[doc_id]}")
            
    except ValueError as e:
        print(f"Configuration or Value Error: {e}")
    except Exception as e:
        # Attempt to get logger from the ranker instance if available, else use global logger
        logger_instance = getattr(ranker, 'logger', logging.getLogger(__name__))
        logger_instance.error(f"An unexpected error occurred during testing: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
