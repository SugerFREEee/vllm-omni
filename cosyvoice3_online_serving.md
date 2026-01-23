# CosyVoice3 Online Serving with vLLM-Omni

This document describes how to deploy CosyVoice3 model for online text-to-speech service using vLLM-Omni.

## 🛠️ Prerequisites

1. **CosyVoice3 Model**: Ensure you have the CosyVoice3 model checkpoint ready.
2. **vLLM-Omni**: Install vLLM-Omni with all dependencies.
3. **Stage Configuration**: Create a proper stage configuration file for CosyVoice3.

## 📁 Project Structure

```
CosyVoice/
├── third_party/
│   └── vllm-omni/
│       ├── cosyvoice3_stage_config.yaml  # Stage configuration for CosyVoice3
│       ├── myscripts/
│       │   ├── vllm_server.py           # Current server implementation
│       │   └── vllm_server_async.py      # Async server implementation
│       └── examples/
│           └── online_serving/
│               └── cosyvoice3/            # CosyVoice3 online serving example
└── myscripts/
    └── cosy_server.py                     # Client implementation
```

## 🚀 Start Server

### Method 1: Using vllm serve command (Recommended)

```bash
# Basic start
vllm serve /path/to/CosyVoice3 --omni --port 8091 --stage-configs-path cosyvoice3_stage_config.yaml

# Start with cache-dit acceleration
vllm serve /path/to/CosyVoice3 --omni --port 8091 --stage-configs-path cosyvoice3_stage_config.yaml --cache-backend cache_dit
```

### Method 2: Using startup script

Create a startup script `run_server.sh`:

```bash
#!/bin/bash
# CosyVoice3 online serving startup script

MODEL="/path/to/CosyVoice3"
PORT="8091"

 echo "Starting CosyVoice3 server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --stage-configs-path cosyvoice3_stage_config.yaml \
    --cache-backend cache_dit
```

Run the script:

```bash
bash run_server.sh
```

## 📝 Stage Configuration

Create `cosyvoice3_stage_config.yaml`:

```yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      process: true
      devices: "0"
      max_batch_size: 16  # Increase for better concurrency
    engine_args:
      model_stage: "diffusion"
      parallel_config:
        pipeline_parallel_size: 1
        data_parallel_size: 1
        tensor_parallel_size: 1
        sequence_parallel_size: 1
        ulysses_degree: 1
        ring_degree: 1
        cfg_parallel_size: 1
      cache_backend: "cache_dit"
      cache_config:
        Fn_compute_blocks: 8
        Bn_compute_blocks: 0
        residual_diff_threshold: 0.08
        max_warmup_steps: 0
        max_cached_steps: -1
        max_continuous_cached_steps: 10
        num_inference_steps: 10
        enable_taylorseer: false
        taylorseer_order: 1
    final_output: true
    final_output_type: "audio"

# Top-level runtime config
runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 4  # Allow multiple inflight requests per stage
  edges:
```

## 🔧 API Design

### Endpoint: `/v1/chat/completions`

#### Request Format

```json
{
  "model": "/path/to/CosyVoice3",
  "messages": [
    {
      "role": "user",
      "content": "Hello, this is a test message for text-to-speech conversion."
    }
  ],
  "extra_body": {
    "num_inference_steps": 10,
    "speaker_id": 0,
    "speed": 1.0,
    "pitch": 1.0
  }
}
```

#### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "/path/to/CosyVoice3",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "audio",
        "audio": {
          "url": "data:audio/wav;base64,...",
          "format": "wav"
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 100,
    "total_tokens": 110
  }
}
```

## 📱 Client Implementation

### Python Client

Create `cosyvoice3_client.py`:

```python
#!/usr/bin/env python3
"""
CosyVoice3 online serving client
"""

import argparse
import base64
import json
import requests


def generate_speech(prompt, output_file, api_url="http://localhost:8091/v1/chat/completions"):
    """Generate speech from text using CosyVoice3 online service"""
    payload = {
        "model": "/path/to/CosyVoice3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "extra_body": {
            "num_inference_steps": 10
        }
    }

    response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    data = response.json()

    # Extract audio from response
    audio_content = data["choices"][0]["message"]["content"][0]["audio"]["url"]
    audio_data = base64.b64decode(audio_content.split(",")[1])

    # Save to file
    with open(output_file, "wb") as f:
        f.write(audio_data)

    print(f"Audio saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice3 online serving client")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for speech generation")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--api-url", type=str, default="http://localhost:8091/v1/chat/completions", help="API endpoint URL")
    args = parser.parse_args()

    generate_speech(args.prompt, args.output, args.api_url)
```

### Usage

```bash
python cosyvoice3_client.py --prompt "Hello, this is a test message" --output output.wav
```

## 📊 Concurrency Testing

### Python Concurrent Test

Create `concurrent_test.py`:

```python
#!/usr/bin/env python3
"""
Concurrent test for CosyVoice3 online service
"""

import argparse
import concurrent.futures
import time
import random
import requests


def test_request(prompt, api_url):
    """Send a single test request"""
    start_time = time.time()
    
    payload = {
        "model": "/path/to/CosyVoice3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        end_time = time.time()
        return True, end_time - start_time
    except Exception as e:
        end_time = time.time()
        return False, end_time - start_time


def run_concurrent_test(api_url, concurrent_count, test_duration):
    """Run concurrent test"""
    prompts = [
        "Hello, this is a test message for text-to-speech conversion.",
        "Welcome to CosyVoice3, your high-quality text-to-speech system.",
        "This is a longer test message to evaluate the system's performance under load.",
        "Testing concurrent requests to measure system throughput and latency.",
        "CosyVoice3 provides natural and expressive speech synthesis capabilities."
    ]

    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    total_response_time = 0.0
    response_times = []

    start_time = time.time()
    end_time = start_time + test_duration

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
        futures = []
        
        while time.time() < end_time:
            prompt = random.choice(prompts)
            future = executor.submit(test_request, prompt, api_url)
            futures.append(future)
            total_requests += 1
            time.sleep(0.01)  # Rate limiting

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            success, response_time = future.result()
            if success:
                successful_requests += 1
                total_response_time += response_time
                response_times.append(response_time)
            else:
                failed_requests += 1

    actual_duration = time.time() - start_time
    if successful_requests > 0:
        avg_response_time = total_response_time / successful_requests
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0
        throughput = successful_requests / actual_duration
    else:
        avg_response_time = 0
        p95_response_time = 0
        throughput = 0

    print("\nConcurrent Test Results:")
    print("=" * 80)
    print(f"Concurrent threads:    {concurrent_count}")
    print(f"Test duration:         {actual_duration:.2f} seconds")
    print(f"Total requests:        {total_requests}")
    print(f"Successful requests:   {successful_requests}")
    print(f"Failed requests:       {failed_requests}")
    print(f"Success rate:          {successful_requests/total_requests*100:.2f}%")
    print(f"Throughput:            {throughput:.2f} requests/second")
    print(f"Average response time: {avg_response_time*1000:.2f} ms")
    print(f"P95 response time:     {p95_response_time*1000:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent test for CosyVoice3 online service")
    parser.add_argument("--api-url", type=str, default="http://localhost:8091/v1/chat/completions", help="API endpoint URL")
    parser.add_argument("--concurrent-count", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--test-duration", type=int, default=60, help="Test duration in seconds")
    args = parser.parse_args()

    run_concurrent_test(args.api_url, args.concurrent_count, args.test_duration)
```

### Usage

```bash
# Test with 10 concurrent requests
python concurrent_test.py --concurrent-count 10 --test-duration 60

# Test with 30 concurrent requests
python concurrent_test.py --concurrent-count 30 --test-duration 60
```

## 🔄 Integration with Existing Code

### Modify cosy_server.py

Update `cosy_server.py` to support both the current ZeroMQ-based client and the new HTTP-based client:

```python
# Add HTTP client support
def _run_http_client_test(model_dir, device_id, dtype_name, input_lst, tokens_dir, api_url, concurrent_count, test_duration):
    """Run concurrent test using HTTP API"""
    # Similar to _run_concurrent_test but uses HTTP API instead of ZeroMQ
    pass

# Add command-line argument for API URL
parser.add_argument("--api-url", type=str, default="http://localhost:8091/v1/chat/completions", help="HTTP API endpoint URL")
parser.add_argument("--use-http", action="store_true", help="Use HTTP API instead of ZeroMQ")
```

## 📈 Performance Optimization

### 1. **Batch Processing**
- Increase `max_batch_size` in the stage configuration to allow batching of requests.
- Set `max_inflight` to a higher value to allow multiple inflight requests per stage.

### 2. **Cache Optimization**
- Use `cache_dit` backend with appropriate configuration:
  - `Fn_compute_blocks`: 8 (balance between memory and speed)
  - `residual_diff_threshold`: 0.08 (sensitivity of cache)
  - `max_continuous_cached_steps`: 10 (match num_inference_steps)

### 3. **Parallelism**
- Use sequence parallelism for better GPU utilization:
  - `ulysses_degree`: 1
  - `ring_degree`: 1
  - `sequence_parallel_size`: 1

### 4. **Memory Management**
- Use `float32` for stability (as recommended for CosyVoice3)
- Monitor GPU memory usage and adjust batch size accordingly

## 🚦 Monitoring and Debugging

### Server Logs

```bash
# Start server with verbose logging
vllm serve /path/to/CosyVoice3 --omni --port 8091 --log-level debug
```

### Metrics

The server provides basic metrics through the API response headers:
- `X-Request-ID`: Unique request identifier
- `X-Processing-Time`: Time taken to process the request

### Health Check

```bash
# Health check endpoint
curl http://localhost:8091/health
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model path is correct and accessible.
2. **Stage Configuration Error**: Check the stage configuration file format and parameters.
3. **GPU Out of Memory**: Reduce `max_batch_size` or use a smaller model.
4. **Concurrency Issues**: Ensure the server is properly configured for concurrent requests.

### Debug Commands

```bash
# Check GPU usage
top -p $(pgrep -f vllm)

# Check server logs
tail -f /path/to/server.log

# Test single request
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## 📋 Summary

### Key Changes

1. **Stage Configuration**: Create a proper stage configuration file for CosyVoice3.
2. **Server Start**: Use `vllm serve` command instead of custom server implementation.
3. **API Integration**: Implement HTTP client using OpenAI-compatible API.
4. **Concurrency Testing**: Use Python's ThreadPoolExecutor for concurrent testing.
5. **Performance Optimization**: Adjust batch size, cache settings, and parallelism for better performance.

### Expected Performance

| Concurrency | Throughput (req/s) | Avg Response Time (ms) | P95 Response Time (ms) |
|-------------|-------------------|------------------------|------------------------|
| 10          | ~5-10             | ~1000-2000             | ~2500-3500             |
| 20          | ~8-15             | ~1500-2500             | ~3000-4500             |
| 30          | ~10-20            | ~2000-3000             | ~3500-5000             |

### Benefits

1. **Scalability**: Better support for concurrent requests.
2. **Performance**: Improved throughput and latency with proper optimization.
3. **Maintainability**: Uses standard vllm-omni serving infrastructure.
4. **Compatibility**: OpenAI-compatible API for easy integration.
5. **Flexibility**: Easy to adjust parameters and configurations.

## 🎯 Next Steps

1. **Create Stage Configuration**: Finalize the `cosyvoice3_stage_config.yaml` file.
2. **Test Basic Serving**: Start the server and test with a single request.
3. **Optimize Performance**: Adjust parameters based on performance testing.
4. **Scale Testing**: Test with increasing concurrency levels (10 → 20 → 30).
5. **Integration**: Integrate the new HTTP client into existing codebase.

## 📚 References

- [vLLM-Omni Documentation](https://vllm.ai/docs/omni/index.html)
- [Qwen2.5-Omni Online Serving](examples/online_serving/qwen2_5_omni/README.md)
- [Text-to-Image Online Serving](examples/online_serving/text_to_image/README.md)
- [CosyVoice3 Documentation](https://github.com/FunAudioLLM/CosyVoice)

---

**Note**: This is a conceptual design based on vLLM-Omni's existing capabilities. Some adjustments may be needed based on the specific implementation details of CosyVoice3 and vLLM-Omni.
