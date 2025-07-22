"""
Privacy Budget Sensitivity Analysis Experiment for SPACE Framework - Windows Version
Real energy measurement using Windows power monitoring and local Phi-3.5 model.
"""

import time
import json
import statistics
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import os

from space_pipeline import SPACEPipeline
from cloud_sketch_generator import CloudSketchGenerator
from windows_energy_monitor import WindowsPowerMonitor, LocalModelPowerTester

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama_cpp_python not available. Install with: pip install llama-cpp-python")


@dataclass
class ExperimentResult:
    """Data class for storing individual experiment results with real energy measurements."""
    epsilon: float
    prompt: str
    category: str
    processing_time: float
    energy_consumption_joules: float  # Real energy in Joules
    energy_consumption_wh: float      # Real energy in Watt-hours
    average_power_watts: float        # Average power during processing
    privacy_applied: bool
    entities_detected: int
    entities_protected: int
    original_response: str
    final_response: str
    sketch_quality_score: float
    response_quality_score: float
    privacy_preservation_score: float
    power_breakdown: Dict  # CPU/GPU/RAM power breakdown


class LocalLLMJudge:
    """LLM-based quality assessment using local Phi-3.5 model."""
    
    def __init__(self, model_path: str = "D:/Downloads/Phi-3.5-mini-instruct-Q6_K_L.gguf"):
        """Initialize local LLM judge."""
        self.logger = logging.getLogger("LocalLLMJudge")
        self.model_path = model_path
        self.llm_local = None
        self.power_tester = LocalModelPowerTester(model_path)
        
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama_cpp_python is required for local model")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local Phi-3.5 model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        success = self.power_tester.initialize_model()
        if success:
            self.llm_local = self.power_tester.llm_local
            self.logger.info("Local Phi-3.5 judge initialized successfully")
        else:
            raise RuntimeError("Failed to initialize local model")
    
    def evaluate_sketch_quality(self, original_prompt: str, sketch: str, category: str) -> Tuple[float, Dict]:
        """
        Evaluate sketch quality using local model with power monitoring.
        
        Returns:
            Tuple of (quality_score, power_consumption)
        """
        evaluation_prompt = f"""<|system|>
You are an expert evaluator. Rate the quality of semantic sketches on a scale of 0-10.

<|user|>
Evaluate this semantic sketch quality (0-10 scale):

Original Prompt: "{original_prompt}"
Category: {category}  
Generated Sketch: "{sketch}"

Criteria:
1. Structure (0-3): Well-organized and logical?
2. Completeness (0-3): Covers main aspects?
3. Clarity (0-2): Clear and coherent?
4. Appropriateness (0-2): Fits the category?

Provide only a single number score (0-10). No explanation.

<|assistant|>
"""
        
        try:
            response_text, power_results = self.power_tester.test_model_inference(evaluation_prompt)
            
            # Extract numerical score
            score_text = response_text.strip().split()[0]
            score = float(score_text)
            score = max(0, min(10, score))  # Clamp to 0-10
            
            return score, power_results
            
        except Exception as e:
            self.logger.warning(f"Local LLM evaluation failed: {e}")
            return 5.0, {"energy_joules": 0.0, "avg_power_watts": 0.0, "duration_seconds": 0.0}
    
    def evaluate_response_quality(self, original_prompt: str, final_response: str, category: str) -> Tuple[float, Dict]:
        """
        Evaluate response quality using local model with power monitoring.
        
        Returns:
            Tuple of (quality_score, power_consumption)
        """
        evaluation_prompt = f"""<|system|>
You are an expert evaluator. Rate response quality on a scale of 0-10.

<|user|>
Evaluate how well this response addresses the request (0-10 scale):

Original Request: "{original_prompt}"
Category: {category}
Response: "{final_response[:400]}..."

Criteria:
1. Relevance (0-3): Addresses the request?
2. Completeness (0-3): Adequate information?
3. Usefulness (0-2): Helpful to user?
4. Coherence (0-2): Well-organized?

Provide only a single number score (0-10). No explanation.

<|assistant|>
"""
        
        try:
            response_text, power_results = self.power_tester.test_model_inference(evaluation_prompt)
            
            # Extract numerical score
            score_text = response_text.strip().split()[0]
            score = float(score_text)
            score = max(0, min(10, score))  # Clamp to 0-10
            
            return score, power_results
            
        except Exception as e:
            self.logger.warning(f"Local LLM evaluation failed: {e}")
            return 5.0, {"energy_joules": 0.0, "avg_power_watts": 0.0, "duration_seconds": 0.0}
    
    def evaluate_privacy_preservation(self, original_prompt: str, cloud_input: str, entities: List) -> float:
        """Simple privacy preservation evaluation (no LLM needed)."""
        entity_texts = [entity[3] for entity in entities] if entities else []
        
        if not entity_texts:
            return 10.0  # No entities to protect
        
        # Check how many original entities are still visible
        entities_leaked = sum(1 for entity_text in entity_texts if entity_text.lower() in cloud_input.lower())
        leak_ratio = entities_leaked / len(entity_texts)
        
        privacy_score = 10.0 * (1.0 - leak_ratio)
        return privacy_score


class WindowsPrivacyBudgetExperiment:
    """
    Privacy Budget Sensitivity Analysis with Real Windows Energy Monitoring.
    """
    
    def __init__(self, 
                 model_path: str = "D:/Downloads/Phi-3.5-mini-instruct-Q6_K_L.gguf",
                 api_key: str = "sk-Cfwunl2L9eVqjaXXLHfYzg2Od6MEhHjdTQk7xlEegVnZGX6l",
                 base_url: str = "https://api.chatanywhere.tech/v1"):
        """Initialize Windows-based experiment framework."""
        self.logger = logging.getLogger("WindowsPrivacyBudgetExperiment")
        
        # Privacy budget values to test
        self.epsilon_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        # Initialize components
        self.local_judge = LocalLLMJudge(model_path) if LLAMA_CPP_AVAILABLE else None
        self.power_monitor = WindowsPowerMonitor()
        
        # Test dataset
        self.test_prompts = self._load_test_prompts()
        
        self.logger.info(f"Initialized Windows experiment with {len(self.epsilon_values)} ε values and {len(self.test_prompts)} prompts")
        self.logger.info(f"Using local model: {model_path}")
    
    def _load_test_prompts(self) -> List[Tuple[str, str]]:
        """Load test prompts for different categories."""
        return [
            # Tourism category
            ("I want to plan a trip to Tokyo for 3 days with my family", "Tourism"),
            ("My summer holiday is coming, I want to go to Paris for traveling", "Tourism"), 
            ("I would like to spend 4 days exploring New York with 2 family members", "Tourism"),
            
            # Medical category
            ("A 28-year-old female patient named Emma reports symptoms: focal_neurological_symptoms", "Medical"),
            ("My name is Alice and I live in Boston. Find me a doctor.", "Medical"),
            ("A 42-year-old patient reports symptoms: altered_appetite, pain, fever", "Medical"),
            
            # Banking category
            ("I want to file a dispute regarding a charge of $10 on my Chase card", "Banking"),
            ("Dear PNC Bank, I wish to raise a concern regarding a charge of $25 on my account", "Banking"),
            ("I need help with transferring money from my account to another bank", "Banking"),
            
            # Common category
            ("Compose an engaging travel blog post about a recent trip to Hawaii", "Common"),
            ("Draft a professional email seeking feedback on the Quarterly Sales Report", "Common"),
            ("Compare two popular smartphone models for a blog post", "Common")
        ]
    
    def run_single_experiment(self, epsilon: float, prompt: str, category: str) -> ExperimentResult:
        """
        Run single experiment with real energy monitoring.
        
        Args:
            epsilon: Privacy budget value
            prompt: Test prompt
            category: Prompt category
            
        Returns:
            Experiment result with real energy measurements
        """
        self.logger.info(f"Running experiment: ε={epsilon}, category={category}")
        
        # Initialize SPACE pipeline
        pipeline = SPACEPipeline(epsilon_total=epsilon)
        
        # Start power monitoring
        self.power_monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Run end-to-end processing
            result = pipeline.process_prompt_end_to_end(prompt)
            
            # Stop power monitoring and get energy consumption
            processing_time = time.time() - start_time
            power_results = self.power_monitor.stop_monitoring()
            
            # Extract metrics from pipeline result
            if result['status'] == 'success':
                entities_detected = len(result['edge_detection']['entities'])
                entities_protected = len(result['edge_detection']['protected_entities'])
                privacy_applied = result['privacy_preserved']
                cloud_input = result['privacy_protection']['cloud_input']
                sketch = result['cloud_processing']['semantic_sketch']
                final_response = result['edge_refinement']['final_response']
                
                # Evaluate quality using local LLM judge
                if self.local_judge:
                    sketch_quality, sketch_power = self.local_judge.evaluate_sketch_quality(prompt, sketch, category)
                    response_quality, response_power = self.local_judge.evaluate_response_quality(prompt, final_response, category)
                    
                    # Add judge power consumption to total
                    judge_energy = sketch_power.get('energy_joules', 0) + response_power.get('energy_joules', 0)
                    power_results['energy_joules'] += judge_energy
                    power_results['energy_wh'] += judge_energy / 3600.0
                else:
                    sketch_quality = response_quality = 5.0
                
                privacy_score = self.local_judge.evaluate_privacy_preservation(
                    prompt, cloud_input, result['edge_detection']['entities']
                ) if self.local_judge else 5.0
                
            else:
                # Handle error cases
                entities_detected = entities_protected = 0
                privacy_applied = False
                sketch_quality = response_quality = privacy_score = 0.0
                final_response = f"Error: {result.get('error_message', 'Unknown error')}"
            
            return ExperimentResult(
                epsilon=epsilon,
                prompt=prompt,
                category=category,
                processing_time=processing_time,
                energy_consumption_joules=power_results.get('energy_joules', 0.0),
                energy_consumption_wh=power_results.get('energy_wh', 0.0),
                average_power_watts=power_results.get('avg_power_watts', 0.0),
                privacy_applied=privacy_applied,
                entities_detected=entities_detected,
                entities_protected=entities_protected,
                original_response=prompt,
                final_response=final_response,
                sketch_quality_score=sketch_quality,
                response_quality_score=response_quality,
                privacy_preservation_score=privacy_score,
                power_breakdown=power_results.get('power_breakdown', {})
            )
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            # Still get power results for failed experiments
            power_results = self.power_monitor.stop_monitoring()
            
            return ExperimentResult(
                epsilon=epsilon, prompt=prompt, category=category,
                processing_time=time.time() - start_time,
                energy_consumption_joules=power_results.get('energy_joules', 0.0),
                energy_consumption_wh=power_results.get('energy_wh', 0.0),
                average_power_watts=power_results.get('avg_power_watts', 0.0),
                privacy_applied=False, entities_detected=0, entities_protected=0,
                original_response=prompt, final_response=f"Error: {e}",
                sketch_quality_score=0, response_quality_score=0, privacy_preservation_score=0,
                power_breakdown={}
            )
    
    def run_full_experiment(self) -> Dict:
        """Run complete privacy budget sensitivity analysis with real energy monitoring."""
        self.logger.info("Starting Windows privacy budget sensitivity analysis with real energy monitoring")
        all_results = []
        
        total_experiments = len(self.epsilon_values) * len(self.test_prompts)
        experiment_count = 0
        
        for epsilon in self.epsilon_values:
            self.logger.info(f"Testing ε = {epsilon}")
            
            for prompt, category in self.test_prompts:
                experiment_count += 1
                self.logger.info(f"Experiment {experiment_count}/{total_experiments}: {category} - {prompt[:50]}...")
                
                try:
                    result = self.run_single_experiment(epsilon, prompt, category)
                    all_results.append(result)
                    
                    self.logger.info(f"Completed: time={result.processing_time:.3f}s, "
                                   f"energy={result.energy_consumption_joules:.2f}J, "
                                   f"power={result.average_power_watts:.1f}W, "
                                   f"quality={result.response_quality_score:.1f}/10")
                    
                except Exception as e:
                    self.logger.error(f"Experiment failed: {e}")
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Save results
        self._save_results(all_results, analysis)
        
        self.logger.info("Windows privacy budget sensitivity analysis completed")
        return {
            "raw_results": all_results,
            "analysis": analysis,
            "summary": self._generate_summary(analysis)
        }
    
    def _analyze_results(self, results: List[ExperimentResult]) -> Dict:
        """Analyze experimental results with real energy metrics."""
        analysis = {}
        
        for epsilon in self.epsilon_values:
            epsilon_results = [r for r in results if r.epsilon == epsilon]
            
            if not epsilon_results:
                continue
            
            # Calculate aggregate metrics including real energy
            analysis[epsilon] = {
                "experiment_count": len(epsilon_results),
                "success_rate": len([r for r in epsilon_results if r.response_quality_score > 0]) / len(epsilon_results),
                
                # Performance metrics
                "avg_processing_time": statistics.mean([r.processing_time for r in epsilon_results]),
                "std_processing_time": statistics.stdev([r.processing_time for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                
                # Real energy metrics
                "avg_energy_joules": statistics.mean([r.energy_consumption_joules for r in epsilon_results]),
                "std_energy_joules": statistics.stdev([r.energy_consumption_joules for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                "avg_energy_wh": statistics.mean([r.energy_consumption_wh for r in epsilon_results]),
                "avg_power_watts": statistics.mean([r.average_power_watts for r in epsilon_results]),
                "std_power_watts": statistics.stdev([r.average_power_watts for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                
                # Quality metrics
                "avg_sketch_quality": statistics.mean([r.sketch_quality_score for r in epsilon_results]),
                "avg_response_quality": statistics.mean([r.response_quality_score for r in epsilon_results]),
                "avg_privacy_preservation": statistics.mean([r.privacy_preservation_score for r in epsilon_results]),
                
                # Privacy metrics
                "privacy_application_rate": len([r for r in epsilon_results if r.privacy_applied]) / len(epsilon_results),
                "avg_entities_detected": statistics.mean([r.entities_detected for r in epsilon_results]),
                "avg_entities_protected": statistics.mean([r.entities_protected for r in epsilon_results]),
                
                # Energy efficiency metrics
                "energy_per_token_estimate": statistics.mean([r.energy_consumption_joules / max(1, len(r.final_response)) for r in epsilon_results]),
                "power_efficiency_score": statistics.mean([r.response_quality_score / max(1, r.average_power_watts) for r in epsilon_results]),
            }
        
        return analysis
    
    def _save_results(self, results: List[ExperimentResult], analysis: Dict):
        """Save results with real energy measurements."""
        # Convert results to serializable format
        results_data = []
        for result in results:
            results_data.append({
                "epsilon": result.epsilon,
                "prompt": result.prompt,
                "category": result.category,
                "processing_time": result.processing_time,
                "energy_consumption_joules": result.energy_consumption_joules,
                "energy_consumption_wh": result.energy_consumption_wh,
                "average_power_watts": result.average_power_watts,
                "privacy_applied": result.privacy_applied,
                "entities_detected": result.entities_detected,
                "entities_protected": result.entities_protected,
                "sketch_quality_score": result.sketch_quality_score,
                "response_quality_score": result.response_quality_score,
                "privacy_preservation_score": result.privacy_preservation_score,
                "final_response_length": len(result.final_response),
                "power_breakdown": result.power_breakdown
            })
        
        # Save to JSON
        output_file = '/Users/zhanjunfei/cursor/space/space_ce/windows_privacy_budget_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                "raw_results": results_data,
                "analysis": analysis,
                "metadata": {
                    "total_experiments": len(results_data),
                    "epsilon_values": self.epsilon_values,
                    "energy_unit": "Joules (real measurement)",
                    "power_unit": "Watts (real measurement)"
                }
            }, f, indent=2)
        
        # Save to CSV
        df = pd.DataFrame(results_data)
        csv_file = '/Users/zhanjunfei/cursor/space/space_ce/windows_privacy_budget_results.csv'
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {output_file} and {csv_file}")
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate summary with real energy metrics."""
        summary = "Windows Privacy Budget Sensitivity Analysis - Real Energy Measurement\n"
        summary += "=" * 70 + "\n\n"
        
        summary += "Performance vs Privacy Budget (Real Energy Consumption):\n"
        summary += f"{'ε Value':<10} {'Avg Time':<12} {'Avg Energy':<15} {'Avg Power':<12} {'Response Quality':<16}\n"
        summary += "-" * 80 + "\n"
        
        for epsilon in self.epsilon_values:
            if epsilon in analysis:
                stats = analysis[epsilon]
                summary += f"{epsilon:<10} {stats['avg_processing_time']:<12.3f} "
                summary += f"{stats['avg_energy_joules']:<15.2f} {stats['avg_power_watts']:<12.1f} "
                summary += f"{stats['avg_response_quality']:<16.1f}\n"
        
        summary += f"\nEnergy Units: Joules (J) - Real measured energy consumption\n"
        summary += f"Power Units: Watts (W) - Real measured power consumption\n"
        
        return summary
    
    def cleanup(self):
        """Clean up resources."""
        self.power_monitor.cleanup()
        if self.local_judge:
            self.local_judge.power_tester.cleanup()


def main():
    """Run Windows privacy budget sensitivity analysis with real energy monitoring."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    print("🔋 Windows Privacy Budget Sensitivity Analysis - Real Energy Monitoring")
    print("=" * 80)
    print("Features:")
    print("- Real Windows power consumption measurement (CPU/GPU/RAM)")
    print("- Local Phi-3.5 model for quality evaluation")
    print("- Energy consumption in Joules (not simulated)")
    print("- Privacy budget sensitivity testing: ε = {0.01, 0.1, 1, 10, 100}")
    print("-" * 80)
    
    # Check requirements
    if not LLAMA_CPP_AVAILABLE:
        print("❌ Error: llama_cpp_python not available")
        print("Install with: pip install llama-cpp-python")
        return
    
    model_path = "D:/Downloads/Phi-3.5-mini-instruct-Q6_K_L.gguf"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please download Phi-3.5-mini-instruct-Q6_K_L.gguf to the specified path")
        return
    
    # Initialize and run experiment
    experiment = WindowsPrivacyBudgetExperiment(model_path=model_path)
    
    try:
        results = experiment.run_full_experiment()
        
        # Display summary
        print("\n" + results["summary"])
        
        print("\n✅ Windows privacy budget sensitivity analysis completed!")
        print("📊 Results with REAL energy measurements saved to:")
        print("   - windows_privacy_budget_results.json")
        print("   - windows_privacy_budget_results.csv")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()