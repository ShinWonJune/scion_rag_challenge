# main_script.py
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import json
import os
import re
import sys
import time
import random
import concurrent.futures
from functools import partial

# --- Helper Functions (Unchanged) ---


def _extract_json(text: str) -> Any:
    """Extract a JSON object/array from a model response.

    Handles fenced blocks and stray prose. Raises ValueError on failure.
    """
    if text is None:
        raise ValueError("Empty response from model.")
    # Try code fences first
    fence = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    # Trim leading/trailing non-json
    start = candidate.find("{")
    start_arr = candidate.find("[")
    if start == -1 or (start_arr != -1 and start_arr < start):
        start = start_arr
    if start == -1:
        raise ValueError("No JSON object/array found in response.")
    # Find last closing brace/bracket
    end = max(candidate.rfind("}"), candidate.rfind("]")) + 1
    payload = candidate[start:end]
    return json.loads(payload)


def _retry_sleep(base: float, attempt: int, jitter: bool = True) -> None:
    # Exponential backoff with jitter
    delay = base * (2 ** (attempt - 1))
    if jitter:
        delay = delay * (0.5 + random.random())
    time.sleep(min(delay, 30.0))


# --- Single API Call Function (Unchanged, used as a worker) ---


def call_gemini(
    model_obj: Any, messages: List[Dict[str, str]], max_retries: int = 3
) -> Any:
    """
    Makes a single call to the Gemini API with retry logic.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # Assuming the content to send is always the first part of the first message
            resp = model_obj.generate_content(messages[0]["parts"][0])
            text = resp.text if hasattr(resp, "text") else str(resp)
            return _extract_json(text)
        except Exception as e:  # includes parsing and API errors
            last_err = e
            print(f"Attempt {attempt} failed for a request: {e}")
            if attempt == max_retries:
                break
            _retry_sleep(0.7, attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# --- ✨ New Parallel API Call Function ✨ ---


def call_gemini_parallel(
    model_obj: Any,
    list_of_messages: List[List[Dict[str, str]]],
    max_workers: int,
    max_retries: int = 3,
) -> List[Any]:
    """
    Calls the Gemini API in parallel for a list of different requests.

    Args:
        model_obj (Any): The initialized Gemini model object.
        list_of_messages (List[List[Dict[str, str]]]): A list where each element
            is a 'messages' list for a single, independent API call.
        max_workers (int): The maximum number of parallel API calls to run at once.
        max_retries (int): The maximum number of retries for each individual call.

    Returns:
        List[Any]: A list of parsed JSON responses in the same order as the input.
                   If any call fails after all retries, this function will
                   raise the exception from that failed call.
    """
    if not list_of_messages:
        print("Warning: No messages provided for parallel processing.")
        return []

    # functools.partial creates a new function with some arguments of the
    # original function already filled in. This is a clean way to pass
    # the static 'model_obj' and 'max_retries' arguments to the worker threads.
    worker_func = partial(call_gemini, model_obj=model_obj, max_retries=max_retries)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applies the worker_func to each item in list_of_messages.
        # It automatically manages the threads and returns results in the same order.
        try:
            # We wrap it in list() to ensure all tasks are completed before moving on.
            results = list(executor.map(worker_func, list_of_messages))
        except Exception as e:
            print(f"An error occurred in one of the parallel API calls: {e}")
            # The exception from the first failed job is raised by executor.map
            raise

    return results


# --- Example Usage ---

if __name__ == "__main__":
    # This is a mock setup. Replace with your actual genai.configure and model initialization.
    # try:
    #     genai.configure(api_key="YOUR_API_KEY")
    #     model = genai.GenerativeModel('gemini-pro')
    # except Exception as e:
    #     print(f"Skipping example run: Gemini not configured. Error: {e}")
    #     sys.exit(0)

    # --- Mock Model for demonstration without a real API key ---
    class MockResponse:
        def __init__(self, text):
            self.text = text

    class MockModel:
        def generate_content(self, content):
            print(f"  (Mocking API call for: '{content[:30]}...')")
            time.sleep(0.5)  # Simulate network latency
            # Simulate a JSON response
            response_data = {"status": "success", "input_received": content}
            return MockResponse(f"```json\n{json.dumps(response_data)}\n```")

    model = MockModel()
    # -----------------------------------------------------------

    # 1. Prepare a list of requests to be sent in parallel.
    # Each item in the list is the 'messages' argument for one call.
    all_requests = [
        [{"parts": ["What is the capital of France?"]}],
        [{"parts": ["Summarize the theory of relativity."]}],
        [{"parts": ["Who wrote 'To Kill a Mockingbird'?"]}],
        [{"parts": ["What are the main components of a CPU?"]}],
        [{"parts": ["Explain the difference between TCP and UDP."]}],
    ]

    print(f"Sending {len(all_requests)} requests in parallel...")

    # 2. Set the maximum number of parallel workers.
    # Your effective throughput will be limited by this number, your network,
    # and the API's rate limits.
    MAX_CONCURRENT_REQUESTS = 3

    # 3. Call the parallel function.
    start_time = time.time()
    try:
        parallel_results = call_gemini_parallel(
            model_obj=model,
            list_of_messages=all_requests,
            max_workers=MAX_CONCURRENT_REQUESTS,
        )
        end_time = time.time()

        # 4. Print the results.
        print("\n--- All parallel requests completed ---")
        for i, result in enumerate(parallel_results):
            print(f"Response for request {i + 1}:")
            print(json.dumps(result, indent=2))
            print("-" * 20)

        print(f"\n✅ Successfully completed {len(parallel_results)} requests.")
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"\n❌ The parallel execution failed. Reason: {e}")
