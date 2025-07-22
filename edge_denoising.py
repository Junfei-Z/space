"""
Edge-Side Denoising and Reintegration for SPACE Framework
Implements sketch refinement and privacy-preserving response synthesis.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional


class EdgeDenoisingReintegrator:
    """
    Edge-side denoising and reintegration component.
    Refines cloud-generated sketches with original context and entities.
    """
    
    def __init__(self):
        """Initialize edge denoising reintegrator."""
        self.logger = logging.getLogger("EdgeDenoisingReintegrator")
        
        # Entity placeholder patterns for different obfuscation formats
        self.placeholder_patterns = [
            r'\[([A-Z_]+)\]_(\w+)',  # [PERSON]_Alice format
            r'\[([A-Z_]+)\]',        # [PERSON] format
            r'\[MASKED_([A-Z_]+)\]', # [MASKED_PERSON] format
            r'\[XXX\]',              # Generic mask
            r'<([A-Z_]+)>',          # <PERSON> format
        ]
        
        # Category-specific refinement strategies
        self.refinement_strategies = {
            "Tourism": self._refine_tourism_sketch,
            "Medical": self._refine_medical_sketch,
            "Banking": self._refine_banking_sketch,
            "Common": self._refine_common_sketch
        }
    
    def refine_sketch(self, 
                     sketch: str,
                     original_prompt: str,
                     entities: List[Tuple[str, int, int, str]],
                     protected_entities: List[Tuple[str, int, int, str]],
                     category: str) -> str:
        """
        Complete edge-side denoising and reintegration.
        
        Args:
            sketch: Cloud-generated semantic sketch
            original_prompt: Original user prompt
            entities: All detected entities
            protected_entities: Entities that were privacy-protected
            category: Detected category
            
        Returns:
            Final refined response
        """
        self.logger.info(f"Refining {category} sketch with {len(protected_entities)} protected entities")
        
        # Step 1: Replace obfuscated entities with originals
        denoised_sketch = self._replace_obfuscated_entities(
            sketch, original_prompt, entities, protected_entities
        )
        
        # Step 2: Apply category-specific refinement
        refinement_func = self.refinement_strategies.get(category, self._refine_common_sketch)
        refined_response = refinement_func(
            denoised_sketch, original_prompt, entities, protected_entities
        )
        
        # Step 3: Post-processing and quality checks
        final_response = self._post_process_response(
            refined_response, original_prompt, category
        )
        
        self.logger.info(f"Refinement complete: {len(final_response)} characters")
        return final_response
    
    def _replace_obfuscated_entities(self, 
                                   sketch: str,
                                   original_prompt: str,
                                   entities: List[Tuple[str, int, int, str]],
                                   protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """
        Replace obfuscated entities in sketch with original values.
        
        Args:
            sketch: Cloud-generated sketch
            original_prompt: Original prompt
            entities: All detected entities
            protected_entities: Protected entities
            
        Returns:
            Sketch with obfuscated entities replaced
        """
        denoised_sketch = sketch
        
        # Create entity mapping: type -> original values
        entity_mapping = {}
        for entity_type, start, end, entity_text in protected_entities:
            if entity_type not in entity_mapping:
                entity_mapping[entity_type] = []
            entity_mapping[entity_type].append(entity_text)
        
        # Replace placeholders with original entities
        for pattern in self.placeholder_patterns:
            matches = re.finditer(pattern, denoised_sketch)
            
            for match in matches:
                full_match = match.group(0)
                
                if len(match.groups()) >= 2:
                    # Format: [TYPE]_value
                    entity_type = match.group(1)
                    obfuscated_value = match.group(2)
                elif len(match.groups()) == 1:
                    # Format: [TYPE] or <TYPE>
                    entity_type = match.group(1)
                    obfuscated_value = None
                else:
                    # Generic [XXX]
                    entity_type = None
                    obfuscated_value = None
                
                # Find replacement
                replacement = self._find_replacement_entity(
                    entity_type, obfuscated_value, entity_mapping, original_prompt
                )
                
                if replacement:
                    denoised_sketch = denoised_sketch.replace(full_match, replacement, 1)
                    self.logger.debug(f"Replaced {full_match} -> {replacement}")
        
        return denoised_sketch
    
    def _find_replacement_entity(self, 
                               entity_type: Optional[str],
                               obfuscated_value: Optional[str],
                               entity_mapping: Dict[str, List[str]],
                               original_prompt: str) -> Optional[str]:
        """
        Find the best replacement for an obfuscated entity.
        
        Args:
            entity_type: Type of entity to replace
            obfuscated_value: Obfuscated value (if any)
            entity_mapping: Mapping of types to original values
            original_prompt: Original prompt for context
            
        Returns:
            Best replacement entity or None
        """
        if entity_type and entity_type in entity_mapping:
            originals = entity_mapping[entity_type]
            
            if originals:
                # If we have multiple options, prefer the first one
                # In a more sophisticated implementation, we could use
                # semantic similarity or position in the original prompt
                return originals[0]
        
        # Fallback: try to find any entity in the original prompt
        # that might be contextually appropriate
        if entity_type:
            # Look for entity type keywords in original prompt
            type_keywords = {
                "PERSON": ["name", "patient", "customer", "user"],
                "LOCATION": ["city", "place", "destination", "location"],
                "ORGANIZATION": ["bank", "hospital", "company", "organization"],
                "DATE_TIME": ["age", "date", "time", "year", "day"],
            }
            
            keywords = type_keywords.get(entity_type, [])
            for keyword in keywords:
                if keyword in original_prompt.lower():
                    # Extract potential entity near the keyword
                    # This is a simplified approach
                    words = original_prompt.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower() and i < len(words) - 1:
                            return words[i + 1]
        
        return None
    
    def _refine_tourism_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine tourism category sketches."""
        
        # Extract key information from original prompt
        duration_info = self._extract_duration(original_prompt)
        location_info = self._extract_locations(entities)
        group_info = self._extract_group_info(original_prompt)
        
        # Build refined response
        refined_response = f"Here's a personalized itinerary for your trip:\n\n"
        
        # Add context from original prompt
        if location_info:
            refined_response += f"**Destination:** {location_info[0]}\n"
        if duration_info:
            refined_response += f"**Duration:** {duration_info}\n"
        if group_info:
            refined_response += f"**Travel Style:** {group_info}\n\n"
        
        # Add the refined sketch
        refined_response += "**Detailed Itinerary:**\n"
        refined_response += sketch
        
        # Add practical tips
        refined_response += "\n\n**Additional Tips:**\n"
        refined_response += "- Book accommodations in advance\n"
        refined_response += "- Check local weather and pack accordingly\n"
        refined_response += "- Keep important documents and emergency contacts handy"
        
        return refined_response
    
    def _refine_medical_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine medical category sketches."""
        
        # Extract medical context
        age_info = self._extract_age(original_prompt)
        symptoms_info = self._extract_symptoms(original_prompt)
        
        # Build refined medical response
        refined_response = "**Clinical Assessment Summary:**\n\n"
        
        if age_info:
            refined_response += f"**Patient Demographics:** {age_info}\n"
        if symptoms_info:
            refined_response += f"**Presenting Symptoms:** {', '.join(symptoms_info)}\n\n"
        
        # Add the clinical sketch
        refined_response += "**Recommended Clinical Approach:**\n"
        refined_response += sketch
        
        # Add medical disclaimers
        refined_response += "\n\n**Important Notes:**\n"
        refined_response += "- This is a general clinical framework for reference\n"
        refined_response += "- Actual diagnosis and treatment should involve qualified medical professionals\n"
        refined_response += "- Emergency cases require immediate medical attention"
        
        return refined_response
    
    def _refine_banking_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine banking category sketches."""
        
        # Extract banking context
        amount_info = self._extract_amounts(original_prompt)
        bank_info = self._extract_banks(entities)
        
        # Build refined banking response
        refined_response = "**Banking Service Request Summary:**\n\n"
        
        if bank_info:
            refined_response += f"**Financial Institution:** {bank_info[0]}\n"
        if amount_info:
            refined_response += f"**Transaction Amount:** {amount_info[0]}\n\n"
        
        # Add the banking sketch
        refined_response += "**Service Process Overview:**\n"
        refined_response += sketch
        
        # Add banking notices
        refined_response += "\n\n**Important Information:**\n"
        refined_response += "- Processing times may vary based on transaction type\n"
        refined_response += "- Keep all transaction receipts and confirmations\n"
        refined_response += "- Contact customer service for urgent issues"
        
        return refined_response
    
    def _refine_common_sketch(self, 
                            sketch: str,
                            original_prompt: str,
                            entities: List[Tuple[str, int, int, str]],
                            protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine common category sketches."""
        
        # Build general refined response
        refined_response = "**Content Creation Framework:**\n\n"
        
        # Add the sketch
        refined_response += "**Structured Approach:**\n"
        refined_response += sketch
        
        # Add general tips
        refined_response += "\n\n**Implementation Tips:**\n"
        refined_response += "- Adapt the structure to your specific needs\n"
        refined_response += "- Consider your target audience\n"
        refined_response += "- Review and refine the content before finalizing"
        
        return refined_response
    
    def _post_process_response(self, 
                             response: str,
                             original_prompt: str,
                             category: str) -> str:
        """
        Final post-processing and quality checks.
        
        Args:
            response: Refined response
            original_prompt: Original prompt
            category: Category
            
        Returns:
            Final processed response
        """
        # Clean up formatting
        response = re.sub(r'\n{3,}', '\n\n', response)  # Remove excessive newlines
        response = response.strip()
        
        # Ensure minimum response quality
        if len(response) < 100:
            # If response is too short, add fallback content
            fallback = self._generate_fallback_response(original_prompt, category)
            response = fallback + "\n\n" + response
        
        # Add category-specific footer if needed
        if "disclaimer" not in response.lower() and category == "Medical":
            response += "\n\n*This information is for reference only and should not replace professional medical advice.*"
        
        return response
    
    def _generate_fallback_response(self, original_prompt: str, category: str) -> str:
        """Generate fallback response if sketch is too short."""
        fallbacks = {
            "Tourism": f"Based on your travel request, here's a comprehensive approach to planning your trip:",
            "Medical": f"Regarding your medical inquiry, here's a structured clinical approach:",
            "Banking": f"For your banking service request, here's the recommended process:",
            "Common": f"Based on your request, here's a structured approach:"
        }
        return fallbacks.get(category, "Here's a structured approach to your request:")
    
    # Helper methods for extracting information
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration information from text."""
        duration_patterns = [
            r'(\d+)\s+days?',
            r'(\d+)\s+nights?',
            r'for\s+(\d+)\s+days?',
            r'(\d+)-day',
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} days"
        return None
    
    def _extract_locations(self, entities: List[Tuple[str, int, int, str]]) -> List[str]:
        """Extract location entities."""
        return [entity_text for entity_type, _, _, entity_text in entities if entity_type == "LOCATION"]
    
    def _extract_group_info(self, text: str) -> Optional[str]:
        """Extract group/travel style information."""
        if "solo" in text.lower():
            return "Solo travel"
        elif "family" in text.lower():
            return "Family travel"
        elif "group" in text.lower():
            return "Group travel"
        elif "couple" in text.lower():
            return "Couple travel"
        return None
    
    def _extract_age(self, text: str) -> Optional[str]:
        """Extract age information."""
        age_pattern = r'(\d+)-year-old'
        match = re.search(age_pattern, text)
        return f"{match.group(1)} years old" if match else None
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptom information."""
        # Simple symptom extraction - could be enhanced
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "fatigue",
            "symptoms", "altered_appetite", "focal_neurological_symptoms"
        ]
        
        found_symptoms = []
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts."""
        amount_pattern = r'\$(\d+(?:\.\d{2})?)'
        matches = re.findall(amount_pattern, text)
        return [f"${amount}" for amount in matches]
    
    def _extract_banks(self, entities: List[Tuple[str, int, int, str]]) -> List[str]:
        """Extract bank/organization entities."""
        banks = []
        for entity_type, _, _, entity_text in entities:
            if entity_type == "ORGANIZATION" and any(word in entity_text.lower() 
                                                   for word in ["bank", "chase", "wells", "pnc"]):
                banks.append(entity_text)
        return banks


def main():
    """Test the edge denoising reintegrator."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize reintegrator
    reintegrator = EdgeDenoisingReintegrator()
    
    print("🔧 SPACE Edge-Side Denoising and Reintegration Demo")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "category": "Tourism",
            "original_prompt": "I want to plan a trip to Tokyo for 3 days with my family.",
            "sketch": "Day 1: Morning – arrival and accommodation setup; Afternoon – city orientation and major landmark visit; Evening – family-friendly dining. Day 2: Morning – cultural/historical site tour; Afternoon – interactive museum or park; Evening – local performance or entertainment. Day 3: Morning – local market or shopping district; Afternoon – scenic outdoor activity; Evening – departure preparation.",
            "entities": [("LOCATION", 25, 30, "Tokyo"), ("DATE_TIME", 35, 41, "3 days")],
            "protected_entities": [("LOCATION", 25, 30, "Tokyo")]
        },
        {
            "category": "Medical", 
            "original_prompt": "A 28-year-old female patient named Emma reports symptoms: focal_neurological_symptoms",
            "sketch": "Patient Assessment: Age and gender noted; Symptom Documentation: Neurological symptoms recorded; Recommended Actions: Neurological examination, imaging studies, specialist consultation; Follow-up: Monitor symptom progression, schedule return visit.",
            "entities": [("DATE_TIME", 2, 13, "28-year-old"), ("PERSON", 34, 38, "Emma")],
            "protected_entities": [("PERSON", 34, 38, "Emma")]
        },
        {
            "category": "Banking",
            "original_prompt": "I want to file a dispute regarding a charge of $10 on my Chase card",
            "sketch": "Dispute Request: Charge amount and card identification; Documentation Needed: Transaction details, supporting evidence; Process Steps: Initial review, investigation period, provisional credit consideration; Timeline: Standard dispute resolution timeframe, status updates.",
            "entities": [("ORGANIZATION", 56, 61, "Chase")],
            "protected_entities": [("ORGANIZATION", 56, 61, "Chase")]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['category']}")
        print(f"Original: {test_case['original_prompt']}")
        print(f"Sketch: {test_case['sketch'][:100]}...")
        print("-" * 50)
        
        refined_response = reintegrator.refine_sketch(
            sketch=test_case['sketch'],
            original_prompt=test_case['original_prompt'], 
            entities=test_case['entities'],
            protected_entities=test_case['protected_entities'],
            category=test_case['category']
        )
        
        print(f"Refined Response:\n{refined_response}")


if __name__ == "__main__":
    main()