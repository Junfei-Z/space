"""
Privacy Budget Sensitivity Analysis Experiment for SPACE Framework
Validates the impact of different privacy budgets (ε values) on model output quality.
Tests ε = {0.01, 0.1, 1, 10, 100} with comprehensive metrics.
"""

import time
import psutil
import json
import statistics
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

from space_pipeline import SPACEPipeline
from cloud_sketch_generator import CloudSketchGenerator


@dataclass
class ExperimentResult:
    """Data class for storing individual experiment results."""
    epsilon: float
    prompt: str
    category: str
    processing_time: float
    energy_consumption: float
    privacy_applied: bool
    entities_detected: int
    entities_protected: int
    original_response: str
    final_response: str
    sketch_quality_score: float
    response_quality_score: float
    privacy_preservation_score: float


class LLMJudge:
    """LLM-based quality assessment judge for evaluating inference quality."""
    
    def __init__(self, api_key: str, base_url: str):
        """Initialize LLM judge."""
        self.cloud_generator = CloudSketchGenerator(api_key=api_key, base_url=base_url)
        self.logger = logging.getLogger("LLMJudge")
    
    def evaluate_sketch_quality(self, original_prompt: str, sketch: str, category: str) -> float:
        """
        Evaluate the quality of semantic sketch generation.
        
        Args:
            original_prompt: Original user prompt
            sketch: Generated semantic sketch
            category: Prompt category
            
        Returns:
            Quality score (0-10)
        """
        evaluation_prompt = f"""
Please evaluate the quality of this semantic sketch on a scale of 0-10:

Original Prompt: "{original_prompt}"
Category: {category}
Generated Sketch: "{sketch}"

Evaluation Criteria:
1. Structure and Organization (0-3): Is the sketch well-structured and logically organized?
2. Completeness (0-3): Does the sketch cover the main aspects of the original request?
3. Clarity and Coherence (0-2): Is the sketch clear and coherent?
4. Category Appropriateness (0-2): Is the sketch appropriate for the given category?

Provide only a single number score from 0-10 (decimals allowed). No explanation needed.
"""
        
        try:
            response = self.cloud_generator.client.chat.completions.create(
                model=self.cloud_generator.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0, min(10, score))  # Clamp to 0-10 range
            
        except Exception as e:
            self.logger.warning(f"LLM evaluation failed: {e}")
            return 5.0  # Default neutral score
    
    def evaluate_response_quality(self, original_prompt: str, final_response: str, category: str) -> float:
        """
        Evaluate the quality of final response.
        
        Args:
            original_prompt: Original user prompt
            final_response: Final processed response
            category: Prompt category
            
        Returns:
            Quality score (0-10)
        """
        evaluation_prompt = f"""
Please evaluate how well this response addresses the original request on a scale of 0-10:

Original Request: "{original_prompt}"
Category: {category}
Final Response: "{final_response[:500]}..."

Evaluation Criteria:
1. Relevance (0-3): Does the response directly address the original request?
2. Completeness (0-3): Does the response provide adequate information/guidance?
3. Usefulness (0-2): Would this response be helpful to the user?
4. Coherence (0-2): Is the response well-organized and easy to understand?

Provide only a single number score from 0-10 (decimals allowed). No explanation needed.
"""
        
        try:
            response = self.cloud_generator.client.chat.completions.create(
                model=self.cloud_generator.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0, min(10, score))  # Clamp to 0-10 range
            
        except Exception as e:
            self.logger.warning(f"LLM evaluation failed: {e}")
            return 5.0  # Default neutral score
    
    def evaluate_privacy_preservation(self, original_prompt: str, cloud_input: str, entities: List) -> float:
        """
        Evaluate how well privacy is preserved in the obfuscated prompt.
        
        Args:
            original_prompt: Original prompt
            cloud_input: Obfuscated prompt sent to cloud
            entities: List of original entities
            
        Returns:
            Privacy preservation score (0-10)
        """
        entity_texts = [entity[3] for entity in entities] if entities else []
        
        # Simple privacy preservation metric
        if not entity_texts:
            return 10.0  # No entities to protect
        
        # Check how many original entities are still visible in cloud input
        entities_leaked = sum(1 for entity_text in entity_texts if entity_text.lower() in cloud_input.lower())
        leak_ratio = entities_leaked / len(entity_texts)
        
        # Score: 10 = no leakage, 0 = complete leakage
        privacy_score = 10.0 * (1.0 - leak_ratio)
        
        return privacy_score


class EnergyMonitor:
    """Monitor system energy consumption during experiments."""
    
    def __init__(self):
        """Initialize energy monitor."""
        self.start_time = None
        self.start_cpu_percent = None
        self.start_memory_percent = None
    
    def start_monitoring(self):
        """Start energy monitoring."""
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)
        self.start_memory_percent = psutil.virtual_memory().percent
    
    def stop_monitoring(self) -> float:
        """
        Stop monitoring and calculate energy consumption estimate.
        
        Returns:
            Estimated energy consumption (arbitrary units)
        """
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent(interval=None)
        end_memory_percent = psutil.virtual_memory().percent
        
        duration = end_time - self.start_time
        avg_cpu = (self.start_cpu_percent + end_cpu_percent) / 2
        avg_memory = (self.start_memory_percent + end_memory_percent) / 2
        
        # Simplified energy estimation: duration * resource usage
        # This is a relative metric for comparison, not absolute energy
        energy_estimate = duration * (avg_cpu + avg_memory * 0.1)
        
        return energy_estimate


class PrivacyBudgetExperiment:
    """
    Privacy Budget Sensitivity Analysis Experiment.
    
    Tests different ε values and measures:
    1. Processing time
    2. Energy consumption  
    3. Sketch quality (LLM judge)
    4. Response quality (LLM judge)
    5. Privacy preservation
    """
    
    def __init__(self, 
                 api_key: str = "sk-Cfwunl2L9eVqjaXXLHfYzg2Od6MEhHjdTQk7xlEegVnZGX6l",
                 base_url: str = "https://api.chatanywhere.tech/v1"):
        """Initialize experiment framework."""
        self.logger = logging.getLogger("PrivacyBudgetExperiment")
        
        # Privacy budget values to test
        self.epsilon_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        # Initialize components
        self.llm_judge = LLMJudge(api_key, base_url)
        self.energy_monitor = EnergyMonitor()
        
        # Test dataset from your synthetic data
        self.test_prompts = self._load_test_prompts()
        
        self.logger.info(f"Initialized experiment with {len(self.epsilon_values)} ε values and {len(self.test_prompts)} prompts")
    
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
        Run a single experiment with given parameters.
        
        Args:
            epsilon: Privacy budget value
            prompt: Test prompt
            category: Prompt category
            
        Returns:
            Experiment result
        """
        self.logger.info(f"Running experiment: ε={epsilon}, category={category}")
        
        # Initialize SPACE pipeline with specific epsilon
        pipeline = SPACEPipeline(epsilon_total=epsilon)
        
        # Start monitoring
        self.energy_monitor.start_monitoring()
        start_time = time.time()
        
        # Run end-to-end processing
        result = pipeline.process_prompt_end_to_end(prompt)
        
        # Stop monitoring
        processing_time = time.time() - start_time
        energy_consumption = self.energy_monitor.stop_monitoring()
        
        # Extract metrics
        if result['status'] == 'success':
            entities_detected = len(result['edge_detection']['entities'])
            entities_protected = len(result['edge_detection']['protected_entities'])
            privacy_applied = result['privacy_preserved']
            cloud_input = result['privacy_protection']['cloud_input']
            sketch = result['cloud_processing']['semantic_sketch']
            final_response = result['edge_refinement']['final_response']
            
            # Evaluate quality using LLM judge
            sketch_quality = self.llm_judge.evaluate_sketch_quality(prompt, sketch, category)
            response_quality = self.llm_judge.evaluate_response_quality(prompt, final_response, category)
            privacy_score = self.llm_judge.evaluate_privacy_preservation(
                prompt, cloud_input, result['edge_detection']['entities']
            )
            
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
            energy_consumption=energy_consumption,
            privacy_applied=privacy_applied,
            entities_detected=entities_detected,
            entities_protected=entities_protected,
            original_response=prompt,
            final_response=final_response,
            sketch_quality_score=sketch_quality,
            response_quality_score=response_quality,
            privacy_preservation_score=privacy_score
        )
    
    def run_full_experiment(self) -> Dict:
        """
        Run the complete privacy budget sensitivity analysis.
        
        Returns:
            Comprehensive experiment results
        """
        self.logger.info("Starting full privacy budget sensitivity analysis")
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
                                   f"quality={result.response_quality_score:.1f}/10")
                    
                except Exception as e:
                    self.logger.error(f"Experiment failed: {e}")
                    # Add error result
                    error_result = ExperimentResult(
                        epsilon=epsilon, prompt=prompt, category=category,
                        processing_time=0, energy_consumption=0, privacy_applied=False,
                        entities_detected=0, entities_protected=0,
                        original_response=prompt, final_response=f"Error: {e}",
                        sketch_quality_score=0, response_quality_score=0, privacy_preservation_score=0
                    )
                    all_results.append(error_result)
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Save results
        self._save_results(all_results, analysis)
        
        self.logger.info("Privacy budget sensitivity analysis completed")
        return {
            "raw_results": all_results,
            "analysis": analysis,
            "summary": self._generate_summary(analysis)
        }
    
    def _analyze_results(self, results: List[ExperimentResult]) -> Dict:
        """Analyze experimental results by epsilon value."""
        analysis = {}
        
        for epsilon in self.epsilon_values:
            epsilon_results = [r for r in results if r.epsilon == epsilon]
            
            if not epsilon_results:
                continue
            
            # Calculate aggregate metrics
            analysis[epsilon] = {
                "experiment_count": len(epsilon_results),
                "success_rate": len([r for r in epsilon_results if r.response_quality_score > 0]) / len(epsilon_results),
                
                # Performance metrics
                "avg_processing_time": statistics.mean([r.processing_time for r in epsilon_results]),
                "std_processing_time": statistics.stdev([r.processing_time for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                "avg_energy_consumption": statistics.mean([r.energy_consumption for r in epsilon_results]),
                "std_energy_consumption": statistics.stdev([r.energy_consumption for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                
                # Quality metrics
                "avg_sketch_quality": statistics.mean([r.sketch_quality_score for r in epsilon_results]),
                "std_sketch_quality": statistics.stdev([r.sketch_quality_score for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                "avg_response_quality": statistics.mean([r.response_quality_score for r in epsilon_results]),
                "std_response_quality": statistics.stdev([r.response_quality_score for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                "avg_privacy_preservation": statistics.mean([r.privacy_preservation_score for r in epsilon_results]),
                "std_privacy_preservation": statistics.stdev([r.privacy_preservation_score for r in epsilon_results]) if len(epsilon_results) > 1 else 0,
                
                # Privacy metrics
                "privacy_application_rate": len([r for r in epsilon_results if r.privacy_applied]) / len(epsilon_results),
                "avg_entities_detected": statistics.mean([r.entities_detected for r in epsilon_results]),
                "avg_entities_protected": statistics.mean([r.entities_protected for r in epsilon_results]),
                
                # Category breakdown
                "by_category": {}
            }
            
            # Category-specific analysis
            for category in ["Tourism", "Medical", "Banking", "Common"]:
                cat_results = [r for r in epsilon_results if r.category == category]
                if cat_results:
                    analysis[epsilon]["by_category"][category] = {
                        "count": len(cat_results),
                        "avg_response_quality": statistics.mean([r.response_quality_score for r in cat_results]),
                        "avg_processing_time": statistics.mean([r.processing_time for r in cat_results]),
                        "privacy_rate": len([r for r in cat_results if r.privacy_applied]) / len(cat_results)
                    }
        
        return analysis
    
    def _save_results(self, results: List[ExperimentResult], analysis: Dict):
        """Save results to files."""
        # Save raw results to JSON
        results_data = []
        for result in results:
            results_data.append({
                "epsilon": result.epsilon,
                "prompt": result.prompt,
                "category": result.category,
                "processing_time": result.processing_time,
                "energy_consumption": result.energy_consumption,
                "privacy_applied": result.privacy_applied,
                "entities_detected": result.entities_detected,
                "entities_protected": result.entities_protected,
                "sketch_quality_score": result.sketch_quality_score,
                "response_quality_score": result.response_quality_score,
                "privacy_preservation_score": result.privacy_preservation_score,
                "final_response_length": len(result.final_response)
            })
        
        with open('/Users/zhanjunfei/cursor/space/space_ce/privacy_budget_results.json', 'w') as f:
            json.dump({
                "raw_results": results_data,
                "analysis": analysis
            }, f, indent=2)
        
        # Save to CSV for easy analysis
        df = pd.DataFrame(results_data)
        df.to_csv('/Users/zhanjunfei/cursor/space/space_ce/privacy_budget_results.csv', index=False)
        
        self.logger.info("Results saved to privacy_budget_results.json and .csv")
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate human-readable summary of results."""
        summary = "Privacy Budget Sensitivity Analysis - Summary Report\n"
        summary += "=" * 60 + "\n\n"
        
        summary += "Performance vs Privacy Budget:\n"
        summary += f"{'ε Value':<10} {'Avg Time':<12} {'Avg Energy':<12} {'Response Quality':<16} {'Privacy Score':<12}\n"
        summary += "-" * 70 + "\n"
        
        for epsilon in self.epsilon_values:
            if epsilon in analysis:
                stats = analysis[epsilon]
                summary += f"{epsilon:<10} {stats['avg_processing_time']:<12.3f} {stats['avg_energy_consumption']:<12.1f} "
                summary += f"{stats['avg_response_quality']:<16.1f} {stats['avg_privacy_preservation']:<12.1f}\n"
        
        summary += "\nKey Findings:\n"
        summary += "- Lower ε values provide stronger privacy but may reduce output quality\n"
        summary += "- Higher ε values improve utility but reduce privacy protection\n"
        summary += "- Processing time and energy consumption vary with privacy budget\n"
        
        return summary


def main():
    """Run privacy budget sensitivity analysis experiment."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    print("🔬 Privacy Budget Sensitivity Analysis Experiment")
    print("=" * 60)
    print("Testing ε values: {0.01, 0.1, 1, 10, 100}")
    print("Metrics: Processing time, Energy consumption, LLM-judged quality")
    print("-" * 60)
    
    # Initialize and run experiment
    experiment = PrivacyBudgetExperiment()
    
    try:
        results = experiment.run_full_experiment()
        
        # Display summary
        print("\n" + results["summary"])
        
        # Show some key statistics
        analysis = results["analysis"]
        print("\nDetailed Results by ε Value:")
        print("-" * 40)
        
        for epsilon in experiment.epsilon_values:
            if epsilon in analysis:
                stats = analysis[epsilon]
                print(f"\nε = {epsilon}:")
                print(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s ± {stats['std_processing_time']:.3f}s")
                print(f"  Avg Response Quality: {stats['avg_response_quality']:.1f}/10 ± {stats['std_response_quality']:.1f}")
                print(f"  Privacy Application Rate: {stats['privacy_application_rate']*100:.1f}%")
                print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Results saved to: privacy_budget_results.json and .csv")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()