import asyncio
import logging
import time

import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Silence vLLM logging
logging.getLogger('vllm').setLevel(logging.CRITICAL)


async def process_request(engine, prompt_tokens, sampling_params, request_id):
    """
    Helper function to process a single request and consume its
    async generator.
    """
    generator = engine.generate(
        TokensPrompt(prompt_token_ids=prompt_tokens), sampling_params, request_id
    )

    final_output = None
    # Iterate through the generator to completion
    async for output in generator:
        final_output = output

    # Return the final output object (contains token_ids, text, etc.)
    return final_output


async def main(args):
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_tokens=args.num_output_tokens,
        max_tokens=args.num_output_tokens,
        seed=args.seed,
    )

    # 1. Create a synthetic prompt of the desired token count
    # Using a list of 0s is a common way to create synthetic input
    # for benchmarking.
    synthetic_prompt_tokens = [0] * args.num_input_tokens

    print(f"Submitting {args.batch_size} synthetic requests...")
    print(f"  - Input token count: {args.num_input_tokens}")
    print(f"  - Output token count: {args.num_output_tokens}")

    torch.cuda.cudart().cudaProfilerStart()

    start_time = time.perf_counter()

    # 2. Create a list of tasks to be run concurrently
    tasks = []
    for i in range(args.batch_size):
        # Create an asyncio task for each request
        task = asyncio.create_task(
            process_request(engine, synthetic_prompt_tokens, sampling_params, str(i))
        )
        tasks.append(task)

    # 3. Wait for all requests to complete concurrently
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    latency = end_time - start_time

    torch.cuda.cudart().cudaProfilerStop()

    print(f"\n--- Benchmark Complete ---")
    print(f"Successfully processed {len(results)} requests in {latency} seconds.")
    print(
        f"Throughput: {args.num_output_tokens * args.batch_size / latency} output tokens / second."
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--batch-size", type=int, required=True, help="Number of concurrent requests to simulate"
    )
    parser.add_argument(
        "--num-input-tokens",
        type=int,
        required=True,
        help="Number of tokens in each synthetic prompt",
    )

    # Keep existing arguments
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    parser.add_argument(
        "--num-output-tokens", type=int, default=32, help="Number of output tokens to generate"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
