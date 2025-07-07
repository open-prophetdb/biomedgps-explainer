import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, mock_open
from drugs4disease.explain import DrugExplain


class TestDrugExplain(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = DrugExplain()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock prompts directory and template file
        self.prompts_dir = os.path.join(self.temp_dir, "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Create mock template file
        self.template_content = """
        Please research the following drugs in relation to {{ disease_name }}:
        
        {% for drug in drugs %}
        - {{ drug }}
        {% endfor %}
        
        Please provide detailed analysis including:
        1. Mechanism of action
        2. Clinical evidence
        3. Safety profile
        4. Potential for repurposing
        """
        
        self.template_file = os.path.join(self.prompts_dir, "drug_deep_research.md")
        with open(self.template_file, 'w') as f:
            f.write(self.template_content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test DrugExplain initialization"""
        explainer = DrugExplain()
        self.assertIsInstance(explainer, DrugExplain)
    
    def test_generate_prompt(self):
        """Test prompt generation"""
        drug_names = ["Aspirin", "Ibuprofen", "Acetaminophen"]
        disease_name = "Alzheimer's Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check if prompt contains expected content
        self.assertIn("Aspirin", prompt)
        self.assertIn("Ibuprofen", prompt)
        self.assertIn("Acetaminophen", prompt)
        self.assertIn("Alzheimer's Disease", prompt)
        self.assertIn("Mechanism of action", prompt)
        self.assertIn("Clinical evidence", prompt)
        self.assertIn("Safety profile", prompt)
        self.assertIn("Potential for repurposing", prompt)
    
    def test_generate_prompt_empty_drug_list(self):
        """Test prompt generation with empty drug list"""
        drug_names = []
        disease_name = "Test Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check if prompt contains disease name but no drugs
        self.assertIn("Test Disease", prompt)
        self.assertNotIn("- ", prompt)  # No drug list items
    
    def test_generate_prompt_single_drug(self):
        """Test prompt generation with single drug"""
        drug_names = ["Aspirin"]
        disease_name = "Heart Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check if prompt contains the single drug
        self.assertIn("Aspirin", prompt)
        self.assertIn("Heart Disease", prompt)
        self.assertEqual(prompt.count("- "), 1)  # Only one drug item
    
    def test_generate_prompt_long_drug_list(self):
        """Test prompt generation with long drug list"""
        drug_names = [f"Drug{i}" for i in range(100)]
        disease_name = "Cancer"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check if prompt contains all drugs
        for i in range(100):
            self.assertIn(f"Drug{i}", prompt)
        
        self.assertIn("Cancer", prompt)
        self.assertEqual(prompt.count("- "), 100)  # All 100 drugs
    
    def test_generate_prompt_special_characters(self):
        """Test prompt generation with special characters in drug names"""
        drug_names = ["Drug-123", "Drug_456", "Drug@789", "Drug#012"]
        disease_name = "Test Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check if all drugs with special characters are included
        for drug in drug_names:
            self.assertIn(drug, prompt)
    
    def test_generate_prompt_template_not_found(self):
        """Test prompt generation when template file is not found"""
        drug_names = ["Aspirin"]
        disease_name = "Test Disease"
        
        # Use a non-existent prompts directory
        with patch('os.path.dirname', return_value="/non/existent/path"):
            with self.assertRaises(FileNotFoundError):
                self.explainer.generate_prompt(
                    drug_names=drug_names,
                    disease_name=disease_name
                )
    
    def test_generate_prompt_invalid_template(self):
        """Test prompt generation with invalid template"""
        # Create invalid template file
        invalid_template = """
        Invalid template with syntax error:
        {% for drug in drugs %}
        - {{ drug }}
        {% endfor %}
        {% if condition %}
        Missing endif
        """
        
        invalid_template_file = os.path.join(self.prompts_dir, "invalid_template.md")
        with open(invalid_template_file, 'w') as f:
            f.write(invalid_template)
        
        drug_names = ["Aspirin"]
        disease_name = "Test Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            with patch('os.path.join', return_value=invalid_template_file):
                with self.assertRaises(Exception):  # Jinja2 will raise an error
                    self.explainer.generate_prompt(
                        drug_names=drug_names,
                        disease_name=disease_name
                    )
    
    def test_generate_prompt_custom_template(self):
        """Test prompt generation with custom template content"""
        custom_template = """
        Research these drugs: {{ drugs|join(', ') }}
        For disease: {{ disease_name }}
        
        Focus on:
        - Efficacy
        - Side effects
        - Cost
        """
        
        custom_template_file = os.path.join(self.prompts_dir, "custom_template.md")
        with open(custom_template_file, 'w') as f:
            f.write(custom_template)
        
        drug_names = ["Drug1", "Drug2", "Drug3"]
        disease_name = "Custom Disease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            with patch('os.path.join', return_value=custom_template_file):
                prompt = self.explainer.generate_prompt(
                    drug_names=drug_names,
                    disease_name=disease_name
                )
        
        # Check if custom template was used
        self.assertIn("Drug1, Drug2, Drug3", prompt)
        self.assertIn("Custom Disease", prompt)
        self.assertIn("Efficacy", prompt)
        self.assertIn("Side effects", prompt)
        self.assertIn("Cost", prompt)
    
    def test_explain_drugs_not_implemented(self):
        """Test that explain_drugs method is not implemented"""
        drug_names = ["Aspirin"]
        disease_name = "Test Disease"
        
        # The method should exist but not be implemented
        result = self.explainer.explain_drugs(
            drug_names=drug_names,
            disease_name=disease_name
        )
        
        # Should return None or raise NotImplementedError
        self.assertIsNone(result)
    
    def test_generate_prompt_with_none_values(self):
        """Test prompt generation with None values"""
        drug_names = None
        disease_name = None
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            with self.assertRaises(Exception):  # Jinja2 will raise an error for None values
                self.explainer.generate_prompt(
                    drug_names=drug_names,
                    disease_name=disease_name
                )
    
    def test_generate_prompt_with_empty_strings(self):
        """Test prompt generation with empty strings"""
        drug_names = ["", "Drug1", ""]
        disease_name = ""
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Should handle empty strings gracefully
        self.assertIn("Drug1", prompt)
        self.assertEqual(prompt.count("- "), 3)  # All three items including empty ones
    
    def test_generate_prompt_template_variables(self):
        """Test that template variables are properly substituted"""
        drug_names = ["TestDrug"]
        disease_name = "TestDisease"
        
        with patch('os.path.dirname', return_value=self.temp_dir):
            prompt = self.explainer.generate_prompt(
                drug_names=drug_names,
                disease_name=disease_name
            )
        
        # Check that template variables are substituted
        self.assertIn("TestDrug", prompt)
        self.assertIn("TestDisease", prompt)
        
        # Check that template syntax is not present in output
        self.assertNotIn("{{", prompt)
        self.assertNotIn("}}", prompt)
        self.assertNotIn("{%", prompt)
        self.assertNotIn("%}", prompt)


if __name__ == '__main__':
    unittest.main() 