import argparse
import asyncio
import time

import aiohttp


async def send_request(session, url, model, prompt_tokens, num_output_tokens, request_id):
    """Send a single completion request and consume the response."""
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": num_output_tokens,
        "min_tokens": num_output_tokens,
        "temperature": 0.3,
        "seed": 42,
    }
    async with session.post(f"{url}/v1/completions", json=payload) as resp:
        result = await resp.json()
        if "error" in result:
            raise RuntimeError(f"Request {request_id} failed: {result['error']}")
        return result


async def main(args):
    synthetic_prompt_tokens = [0] * args.num_input_tokens

    print(f"Submitting {args.batch_size} requests to {args.url}...")
    print(f"  - Input tokens:  {args.num_input_tokens}")
    print(f"  - Output tokens: {args.num_output_tokens}")

    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(
                send_request(
                    session, args.url, args.model,
                    synthetic_prompt_tokens, args.num_output_tokens, i,
                )
            )
            for i in range(args.batch_size)
        ]

        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

    latency = end_time - start_time
    total_output_tokens = args.num_output_tokens * args.batch_size

    print(f"\n--- Benchmark Complete ---")
    print(f"Successfully processed {len(results)} requests in {latency:.2f} seconds.")
    print(f"Throughput: {total_output_tokens / latency:.2f} output tokens / second.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="vLLM server URL")
    parser.add_argument("--model", type=str, required=True, help="Model name (must match server)")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-input-tokens", type=int, required=True)
    parser.add_argument("--num-output-tokens", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
