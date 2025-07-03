import pandas as pd
import re
import os
import jinja2

class DrugExplain:
    """
    Responsible for explaining the drugs.
    """
    def generate_prompt(
        self,
        drug_names: list[str],
        disease_name: str
    ) -> str:
        """
        Generate the prompt for explaining the drugs.
        """
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        template_file = os.path.join(prompts_dir, "drug_deep_research.md")
        with open(template_file, "r") as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            drugs=drug_names,
            disease_name=disease_name,
        )

        return prompt

    def explain_drugs(
        self,
        drug_names: list[str],
        disease_name: str,
    ) -> str:
        """
        Explain the drugs.
        """
        pass
