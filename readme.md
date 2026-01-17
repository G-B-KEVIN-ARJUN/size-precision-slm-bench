# The Tiny Giant SLM Benchmark: Precision vs. Size

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

A rigorous evaluation of Small Language Models (SLMs) designed for edge deployment. This project investigates a critical trade-off in modern AI engineering:


**"Is it better to run a Tiny Model (2B-4B) at High Precision (FP16/INT8), or a Large Model (8B+) at Low Precision (INT4)?"**


This benchmark framework allows developers to scientifically choose the best model for resource-constrained environments (consumer GPUs, laptops, edge devices) by measuring the trade-off between Speed (Throughput) and Intelligence (BERTScore).

## The Hypothesis
In edge AI, memory (VRAM) is the ultimate bottleneck. We often face two architectural choices to fit a model into 8GB VRAM:
1. The "Compressed Giant": Take an 8B+ model (like Llama 3) and compress it down to 4-bit quantization (INT4).
2. The "Native Tiny": Take a specialized 3B model (like Phi-3.5) and run it in high precision (INT8 or FP16).

We test Latency (Speed), Throughput, and Generation Quality to determine which approach yields better real-world performance.

## The Contenders
We tested the industry's leading SLMs (Small Language Models) under 4 Billion parameters.

| Developer | Model Name | Parameters | Precision Tested |
| :--- | :--- | :--- | :--- |
| Microsoft | Phi-3.5 Mini | 3.8B | INT8 (High Precision) |
| Meta | Llama 3.2 | 3.0B | FP16 (Full Precision) |
| Google | Gemma 2 | 2.0B | FP16 (Full Precision) |
| Alibaba | Qwen 2.5 | 3.0B | FP16 (Full Precision) |

## Hardware & Methodology
* GPU: NVIDIA RTX 4060 (8GB VRAM)
* Task: News Summarization (CNN/DailyMail Dataset)
* Metric 1 (Speed): Tokens Per Second (Throughput) & Time To First Token (TTFT).
* Metric 2 (Quality): BERTScore (F1) to measure semantic accuracy against human-written summaries.

## Installation & Usage

1. Clone the Repository
```bash
git clone [https://github.com/your-username/tiny-giant-benchmark.git](https://github.com/your-username/tiny-giant-benchmark.git)
cd tiny-giant-benchmark
```

2. Install Dependencies
Install all required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

3. Download Models
Run the setup script to automatically fetch the specific model versions required for this benchmark. (Note: Ensure you have accepted any necessary licenses on Hugging Face, such as for Llama 3.2)

```bash
python setup_tiny.py
```

4. Run the Benchmark
Execute the main benchmark script. This will load each model sequentially, perform the evaluation tasks, and save the raw performance data to JSON files.

```bash
python benchmark_tiny.py
```

5. Generate Reports
Run the analysis script to process the raw data. This will generate professional metric tables and visualization charts comparing the models.

```bash

python analyze_results.py
```

Output:

benchmark_table_export.png: Detailed metric table with winning stats highlighted.

benchmark_graph_research.png: Scatter plot of Speed vs. Intelligence.

## Extending for Edge Deployment
This framework is built to be Model-Agnostic. It is designed to help teams quickly evaluate new models for edge applications as they are released.

To add a new model (e.g., Mistral, DeepSeek):
```bash
Open benchmark_tiny.py.
```

Add a new entry to the POTENTIAL_MODELS list:

Python
```bash
{ 
    "name": "New-Model-Name",  
    "path": "./models/new-model-folder", 
    "type": "meta"  # Use "google" if it implies strict user/model roles
}
```
Run the benchmark again. The system will automatically detect the new folder, benchmark it, and add it to the comparison graphs.


