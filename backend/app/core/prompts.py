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

    @staticmethod
    def get_claim_extraction_system(version: PromptVersion) -> str:
        """Extract factual claims from a cover letter."""
        if version == PromptVersion.V1:
            return """
            You are an expert fact-checker. Extract ALL factual claims from the provided cover letter.

            A factual claim is any statement that can be verified against objective evidence, including:
            - Job titles, roles, or positions held
            - Companies, organizations, or institutions
            - Dates, durations, or time periods
            - Degrees, certifications, or educational achievements
            - Technologies, tools, frameworks, or programming languages
            - Specific projects, achievements, or accomplishments
            - Quantifiable metrics (team sizes, revenue, performance improvements)
            - Skills or areas of expertise

            DO NOT extract:
            - Generic statements of interest or enthusiasm
            - Subjective opinions or self-assessments
            - Future intentions or desires

            Return STRICT JSON with this format:
            {
                "claims": ["claim1", "claim2", "claim3"]
            }

            Example:
            Input: "I worked as a Senior Engineer at Google for 3 years, leading a team of 5 developers..."
            Output: {
                "claims": [
                    "Worked as a Senior Engineer",
                    "Worked at Google",
                    "Worked for 3 years",
                    "Led a team of 5 developers"
                ]
            }
            """
        else:
            raise ValueError(f"Unsupported prompt version: {version}")

    @staticmethod
    def get_claim_verification_system(version: PromptVersion) -> str:
        """Verify a single claim against the fact table."""
        if version == PromptVersion.V1:
            return """
            You are a rigorous fact-checker. Your job is to verify if a claim from a cover letter
            is supported by the facts extracted from the candidate's CV.

            You will receive:
            1. CLAIM: A specific factual statement from the cover letter
            2. FACTS: The complete fact table extracted from the CV (JSON format)

            Your task:
            - Determine if the CLAIM is directly supported by the FACTS
            - Find the specific fact(s) that support the claim
            - Be STRICT: The claim must be explicitly stated or directly inferable from the facts
            - Do NOT make assumptions or give benefit of the doubt

            Confidence scoring:
            - 1.0: Exact match with explicit fact
            - 0.8-0.9: Strong support, directly inferable
            - 0.5-0.7: Partial support, somewhat related
            - 0.0-0.4: Weak or no support

            Return STRICT JSON with this format:
            {
                "supported": true/false,
                "source": "Description of supporting fact(s) or 'UNSUPPORTED'",
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation of your decision"
            }

            Examples:

            CLAIM: "I worked as a Senior Software Engineer at Google"
            FACTS: [{"category": "experience", "fact": "Senior Software Engineer at Google, 2020-2023"}]
            OUTPUT: {
                "supported": true,
                "source": "CV fact: Senior Software Engineer at Google, 2020-2023",
                "confidence": 1.0,
                "reasoning": "Exact match with CV fact"
            }

            CLAIM: "I led a team of 10 developers"
            FACTS: [{"category": "experience", "fact": "Team Lead managing 5 engineers"}]
            OUTPUT: {
                "supported": false,
                "source": "UNSUPPORTED",
                "confidence": 0.3,
                "reasoning": "CV mentions leading 5 engineers, not 10 developers"
            }
            """
        else:
            raise ValueError(f"Unsupported prompt version: {version}")

