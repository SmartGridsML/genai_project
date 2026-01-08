import json
from enum import Enum
from app.models.schemas import ExtractedFacts, JobAnalysis

class PromptVersion(Enum):
    V1 = "v1"


class Prompts:
    """Centralized prompt repository. Versioned prompts for LLM calls."""

    @staticmethod
    def get_fact_extraction_system(version: PromptVersion) -> str:
        schema_json = json.dumps(ExtractedFacts.model_json_schema(), indent=2)

        if version == PromptVersion.V1:
            return f"""
            You are an expert CV analyst. Extract key facts from the provided text.
            Return STRICT JSON matching the following schema:
            {schema_json}
            """
        else:
            raise ValueError(f"Unsupported prompt version: {version}")

    @staticmethod
    def get_job_description_analysis_system(version: PromptVersion) -> str:
        schema_json = json.dumps(JobAnalysis.model_json_schema(), indent=2)
        if version == PromptVersion.V1:
            return f"""
            Analyze the job description provided. Identify key requirements, 
            culture fit, and technical stacks.
            Return STRICT JSON matching the following schema:
            {schema_json}
            """
        else:
            raise ValueError(f"Unsupported prompt version: {version}")

