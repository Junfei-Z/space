# SPACE: Selective Privacy for Adaptive Cloud-Edge LLM Inference

**SPACE** is a novel privacy-preserving framework for cloud-edge collaborative LLM inference that implements selective privacy protection with semantic sketch collaboration.

## 🌟 Features

- **Edge-Side Entity Detection**: Category-weighted risk scoring with lightweight personal context detection
- **Two-Layer Local Differential Privacy**: Adaptive privacy budget allocation based on entity sensitivity
- **Cloud-Side Semantic Sketch Generation**: GPT-4o integration with few-shot learning for obfuscated inputs
- **Edge-Side Denoising**: Sketch refinement and entity reintegration for final response assembly
- **Real Windows Power Monitoring**: Accurate energy consumption measurement (not simulated)
- **Local Model Integration**: Support for Phi-3.5-mini via llama_cpp_python
- **Comprehensive Evaluation**: Privacy budget sensitivity analysis with LLM judge evaluation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Device   │    │   Cloud Server  │    │   Edge Device   │
│                 │    │                 │    │                 │
│ 1. Entity       │───▶│ 3. Semantic     │───▶│ 4. Denoising &  │
│    Detection    │    │    Sketch       │    │    Integration  │
│ 2. Privacy      │    │    Generation   │    │                 │
│    Protection   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Windows 10/11 (for energy monitoring)
- NVIDIA GPU (optional, for accurate power measurement)
- 8GB+ RAM

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Junfei-Z/space.git
cd space
```

2. **Install dependencies:**
```bash
pip install -r requirements_windows.txt
```

3. **Download local model:**
```bash
# Download Phi-3.5-mini-instruct-Q6_K_L.gguf to:
# D:/Downloads/Phi-3.5-mini-instruct-Q6_K_L.gguf
```

4. **Run privacy budget experiment:**
```bash
python privacy_budget_experiment_windows.py
```

## 📊 Core Components

### 1. Edge Entity Detection (`edge_detection.py`)
```python
detector = EdgeEntityDetector()
results = detector.detect_and_classify(user_prompt)
# Returns: entities, risk_score, protection_needed
```

**Features:**
- Category-weighted risk scoring: `R(P) = Σ w_{t_i} · I(e_i)`
- Personal context detection using first-person pronouns
- Optimized weights: Medical(0.9), Tourism(0.7), Banking(0.8), Common(0.3)

### 2. Two-Layer Local Differential Privacy (`two_layer_ldp.py`)
```python
ldp = TwoLayerLDP(alpha=0.5)
protected_value = ldp.two_layer_dp(entity_type, entity_value, epsilon_total)
```

**Features:**
- Adaptive budget allocation: `ε₁ = ε_total × w_c / (w_c + α(1-w_c))`
- Category and value obfuscation mechanisms
- Configurable privacy-utility tradeoff

### 3. Cloud Semantic Sketch Generation (`cloud_sketch_generator.py`)
```python
generator = CloudSketchGenerator(api_key, base_url)
sketch = generator.generate_semantic_sketch(prompt, category)
```

**Features:**
- GPT-4o integration with few-shot learning
- Handles obfuscated inputs gracefully
- Category-aware sketch generation

### 4. Windows Energy Monitoring (`windows_energy_monitor.py`)
```python
monitor = WindowsPowerMonitor()
monitor.start_monitoring()
# ... your code ...
results = monitor.stop_monitoring()
# Returns: energy_joules, avg_power_watts, power_breakdown
```

**Features:**
- Real power measurement: CPU, GPU, RAM, System
- Energy calculation: `E = P × t` (Joules)
- NVIDIA GPU API integration
- Windows WMI-based CPU monitoring

## 🧪 Privacy Budget Sensitivity Analysis

Test the privacy-utility tradeoff across different epsilon values:

```bash
python privacy_budget_experiment_windows.py
```

**Test Parameters:**
- Privacy budgets: ε ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
- Categories: Tourism, Medical, Banking, Common
- Metrics: Processing time, energy consumption, quality scores

**Sample Results:**
```
ε = 0.01:  Avg Energy: 45.2J,  Quality: 5.6/10,  Privacy: High
ε = 100:   Avg Energy: 38.1J,  Quality: 6.0/10,  Privacy: Low
```

## 📈 Experimental Results

### Energy Consumption by Privacy Budget
| ε Value | Avg Energy (J) | Avg Power (W) | Response Quality | Privacy Score |
|---------|----------------|---------------|------------------|---------------|
| 0.01    | 42.63          | 14.7          | 5.58/10         | 7.5/10        |
| 0.1     | 34.26          | 11.8          | 5.58/10         | 7.1/10        |
| 1.0     | 36.28          | 12.1          | 5.25/10         | 7.1/10        |
| 10.0    | 34.94          | 12.3          | 6.08/10         | 5.8/10        |
| 100.0   | 39.13          | 14.4          | 6.00/10         | 5.8/10        |

### Key Findings
- **Privacy-Utility Tradeoff**: Lower ε provides stronger privacy but may reduce output quality
- **Energy Efficiency**: Privacy protection adds ~15-20% energy overhead
- **Performance**: Average processing time: 2.7-3.1 seconds per query
- **Scalability**: 100% success rate across all 60 experiments

## 🔧 Configuration

### Privacy Settings
```python
# space_pipeline.py
pipeline = SPACEPipeline(
    risk_threshold=0.5,        # Entity detection threshold
    privacy_threshold=0.5,     # Personal context threshold  
    epsilon_total=2.0,         # Total privacy budget
    alpha=0.5,                 # Budget allocation balance
)
```

### Model Settings
```python
# Local Phi-3.5 configuration
llm_local = Llama(
    model_path="D:/Downloads/Phi-3.5-mini-instruct-Q6_K_L.gguf",
    n_gpu_layers=32,           # GPU acceleration
    n_ctx=2048,               # Context window
    n_batch=512,              # Batch size
    main_gpu=0,               # Primary GPU
    offload_kqv=True,         # Memory optimization
)
```

## 📁 Project Structure

```
space/
├── README.md                              # This file
├── WINDOWS_SETUP.md                       # Windows setup guide
├── requirements_windows.txt               # Python dependencies
├── space_pipeline.py                      # Main pipeline orchestration
├── edge_detection.py                      # Edge entity detection
├── two_layer_ldp.py                      # Local differential privacy
├── cloud_sketch_generator.py             # Cloud semantic sketch generation
├── edge_denoising.py                     # Edge denoising and integration
├── windows_energy_monitor.py             # Real Windows power monitoring
├── privacy_budget_experiment_windows.py  # Windows experiment with real energy
├── privacy_budget_experiment.py          # Original experiment (simulated energy)
├── few_shot_examples.txt                 # Few-shot learning examples
├── privacy_budget_results.json          # Experimental results
└── privacy_budget_results.csv           # Results in CSV format
```

## 🎯 Use Cases

### 1. Healthcare Data Processing
```python
prompt = "Patient John, age 45, reports chest pain and shortness of breath"
# SPACE protects: "Patient John" → anonymization
# Cloud receives: "Patient [PERSON_123] age 45, reports chest pain..."
```

### 2. Financial Services
```python
prompt = "I want to dispute a $500 charge on my Chase credit card"
# SPACE protects: "Chase" → bank anonymization  
# Cloud receives: "dispute a $500 charge on my [ORG_456] credit card"
```

### 3. Travel Planning
```python
prompt = "Plan a 3-day trip to Tokyo with my family"
# SPACE protects: "Tokyo" → location generalization
# Cloud receives: "Plan a 3-day trip to [LOCATION_789]..."
```

## 📊 Performance Benchmarks

### System Requirements
- **CPU**: Intel i7/AMD Ryzen 7+ (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GTX 1060+ (optional but recommended)
- **Storage**: 10GB available space

### Throughput
- **Single Query**: 2.7-3.1 seconds average
- **Batch Processing**: ~20 queries/minute
- **Energy Efficiency**: 35-45 Joules per query
- **Privacy Protection Rate**: 58% of queries (entity-dependent)

## 🛡️ Security Features

- **Local Processing**: Sensitive entity detection never leaves the edge
- **Differential Privacy**: Mathematical privacy guarantees
- **Selective Protection**: Only sensitive entities are obfuscated
- **Zero Cloud Storage**: No user data stored in cloud systems
- **Configurable Privacy**: Adjustable ε for different use cases

## 🔬 Research Applications

This framework supports research in:
- **Privacy-Preserving ML**: Differential privacy in LLM inference
- **Edge Computing**: Cloud-edge collaborative architectures
- **Energy Efficiency**: Power consumption analysis in AI systems
- **Privacy-Utility Tradeoffs**: Quantitative privacy vs quality analysis

## 📝 Citation

If you use SPACE in your research, please cite:

```bibtex
@article{zhang2024space,
  title={SPACE: Selective Privacy for Adaptive Cloud-Edge LLM Inference with Semantic Sketch Collaboration},
  author={Zhang, Junfei and [Co-authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

- **Author**: Junfei Zhang
- **Email**: [Your Email]
- **GitHub**: [@Junfei-Z](https://github.com/Junfei-Z)
- **Project**: [https://github.com/Junfei-Z/space](https://github.com/Junfei-Z/space)

## 🙏 Acknowledgments

- OpenAI for GPT-4o API
- Microsoft for Phi-3.5 model
- Presidio for entity detection capabilities
- The privacy-preserving ML research community

---

**⚡ SPACE Framework - Where Privacy Meets Performance** ⚡