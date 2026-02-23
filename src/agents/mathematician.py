import os
import re
import logging
import json
import traceback
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state.agent_state import AgentState
from src.schemas.scribe_schema import VitalsSchema
from src.tools.calculator import calculate_clinical_score
from src.schemas.mathematician_schema import MathematicianSchema
from src.utils.run_with_timeout import run_with_timeout

load_dotenv()

# Logger Configuration
logger = logging.getLogger(__name__)

SEED = 42

# 1. LLM Configuration
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    seed=42,
    num_ctx=8192,
)


# Load Prompt
prompt_path = os.path.join(os.getcwd(), "prompts", "mathematician_prompt.md")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        MATHEMATICIAN_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logger.warning("Mathematician prompt not found. Using default.")
    MATHEMATICIAN_SYSTEM_PROMPT = "You are a clinical mathematician agent."


class MathematicianAgent:
    def __init__(self, model, use_probabilistic: bool = False):
        # Structured model for final analysis
        self.model = model.with_structured_output(MathematicianSchema)
        # Raw model (no structured) for optional probabilistic calc
        self.raw_model = model
        # Create a JSON-mode model for probabilistic scoring (Ollama native JSON mode)
        try:
            self.json_model = model.bind(format="json")
        except Exception:
            self.json_model = model
        env_flag = os.getenv("MATHEMATICIAN_PROBABILISTIC", "").lower() in ("1", "true", "yes")
        self.use_probabilistic = use_probabilistic or env_flag

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        """Robustly extract JSON from LLM output that may contain extra text."""
        # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())

        # 2. Try direct parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 3. Find first { ... } block (greedy, handles nested braces)
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i+1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            break

        # 4. Try fixing common LLM issues: trailing commas, single quotes
        for attempt_text in [cleaned, text]:
            fixed = attempt_text.strip()
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
            fixed = fixed.replace("'", '"')  # single -> double quotes
            # Find JSON block again after fix
            bs = fixed.find("{")
            if bs != -1:
                depth = 0
                for i in range(bs, len(fixed)):
                    if fixed[i] == "{":
                        depth += 1
                    elif fixed[i] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = json.loads(fixed[bs:i+1])
                                if isinstance(data, dict):
                                    return data
                            except json.JSONDecodeError:
                                break

        return {}

    def _calc_scores_with_llm(self, vitals_dict: dict) -> Dict[str, Any]:
        """Pure probabilistic LLM path for calculating NEWS/MEWS.
        Uses Ollama JSON mode + robust parsing. No deterministic fallback —
        this is an experiment to measure the LLM's own scoring ability."""

        # Build a clear prompt with a concrete example (valid JSON, not pseudo-schema)
        example = json.dumps({
            "NEWS": {"score": 5, "risk_level": "Medium", "missing_fields": [], "assumptions": []},
            "MEWS": {"score": 3, "risk_level": "Monitor", "missing_fields": [], "assumptions": []}
        }, indent=2)

        prompt = f"""You are a clinical calculator. Compute NEWS2 and MEWS scores from the vital signs below.
Use the official NEWS2 and MEWS scoring tables. Do NOT invent or assume vitals that are not provided.
If a vital sign is missing (null), assign 0 points for that parameter, list it in "missing_fields", and note the assumption.

Return ONLY valid JSON in this exact format (no extra text):
{example}

Rules:
- "score" must be an integer (the total score), or null if ALL vitals are missing.
- "risk_level" for NEWS: "Low" (0-4), "Medium" (5-6), "High" (>=7).
- "risk_level" for MEWS: "Monitor" (<5), "Critical" (>=5).
- "missing_fields": list of vital sign names that were null/missing.
- "assumptions": list of assumptions made (e.g., "supplemental_oxygen assumed false").

VITALS: {json.dumps(vitals_dict, default=str)}

Respond with JSON only."""

        system_msg = "You are a precise clinical scoring calculator. You MUST respond with valid JSON only, no explanations."

        last_parsed_data = {}

        # Try up to 3 attempts: JSON-mode first, then raw model, then JSON-mode with simplified prompt
        for attempt in range(3):
            try:
                if attempt == 0:
                    model_to_use = self.json_model
                    use_prompt = prompt
                elif attempt == 1:
                    model_to_use = self.raw_model
                    use_prompt = prompt
                else:
                    # Simplified prompt on last attempt
                    model_to_use = self.json_model
                    use_prompt = (
                        f"Calculate NEWS2 and MEWS from these vitals: {json.dumps(vitals_dict, default=str)}\n"
                        f"Missing vitals get 0 points. Return JSON like: {example}"
                    )

                resp = run_with_timeout(
                    model_to_use.invoke,
                    [SystemMessage(content=system_msg),
                     HumanMessage(content=use_prompt)],
                    timeout=120, retries=1
                )
                content = resp.content if hasattr(resp, "content") else str(resp)
                data = self._extract_json_from_text(content)

                if data:
                    last_parsed_data = data
                    # Validate structure: at least one score entry is a dict with 'score' key
                    has_score = any(
                        isinstance(data.get(s), dict) and data[s].get("score") is not None
                        for s in ("NEWS", "MEWS")
                    )
                    if has_score:
                        # Ensure both keys exist with proper structure
                        for s in ("NEWS", "MEWS"):
                            if s not in data or not isinstance(data.get(s), dict):
                                data[s] = {"score": None, "risk_level": "Unknown", "missing_fields": [], "assumptions": ["llm_did_not_compute"]}
                        logger.info(f"LLM scoring succeeded (attempt {attempt+1}): NEWS={data.get('NEWS',{}).get('score')}, MEWS={data.get('MEWS',{}).get('score')}")
                        return data
                    else:
                        logger.warning(f"LLM attempt {attempt+1}: parsed dict but all scores are None.")
                else:
                    logger.warning(f"LLM attempt {attempt+1}: could not extract valid JSON from response.")
                    logger.debug(f"LLM raw output: {content[:500]}")

            except Exception as e:
                logger.warning(f"LLM scoring attempt {attempt+1} error: {e}")

        # All attempts failed — return whatever we last parsed (may have null scores)
        logger.warning("All LLM scoring attempts failed. Returning best-effort result (scores may be null).")
        if last_parsed_data:
            for s in ("NEWS", "MEWS"):
                if s not in last_parsed_data or not isinstance(last_parsed_data.get(s), dict):
                    last_parsed_data[s] = {"score": None, "risk_level": "Unknown", "missing_fields": [], "assumptions": ["llm_all_attempts_failed"]}
            return last_parsed_data
        return {
            "NEWS": {"score": None, "risk_level": "Unknown", "missing_fields": [], "assumptions": ["llm_all_attempts_failed"]},
            "MEWS": {"score": None, "risk_level": "Unknown", "missing_fields": [], "assumptions": ["llm_all_attempts_failed"]}
        }

    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the Mathematician agent node responsible for calculating and analyzing clinical risk scores.

        This function performs the following steps:
        1.  **Data Extraction**: Retrieves vital signs data from the agent state, handling different data structures (objects or dictionaries).
        2.  **Deterministic Calculation**: Uses the `calculate_clinical_score` tool to compute scores like NEWS (National Early Warning Score) and MEWS (Modified Early Warning Score) based on the extracted vitals. It handles potential calculation errors and missing data imputation warnings.
        3.  **LLM Interpretation**: Invokes a structured LLM (Mathematician Model) to analyze the calculated scores, identify any data gaps (estimated scores), and provide an overall risk assessment.
        4.  **Reporting**: Constructs a comprehensive report including raw calculations, analyzed interpretations, and risk assessments to be updated in the agent state.

        Args:
            state (AgentState): The current state of the agent, expected to contain 'extracted_data' with vital signs.

        Returns:
            Dict[str, Any]: A dictionary containing state updates, specifically:
                - 'extracted_data': The original extracted clinical data (preserved).
                - 'risk_score_report': A simple text summary of the calculated scores.
                - 'risk_analysis': A detailed structured analysis including numeric scores, data quality notes, and risk synthesis.
                - In case of error: Returns an error message in 'risk_score_report'.
        """
        logger.info("\n--- 🧮 NODE: MATHEMATICIAN ---")
        # 1. Retrieve vitals with fallback logic

        extracted_data = state.get("extracted_data")
        logger.debug(f"INPUT: {extracted_data}")
        if not extracted_data:
            logger.warning("No extracted data found. Skipping calculation.")
            return {"risk_score": None}
        
        # Robust access to vitals (handles Dict or Pydantic)
        if hasattr(extracted_data, "vitals"):
            vitals = extracted_data.vitals
        elif isinstance(extracted_data, dict):
            vitals = extracted_data.get("vitals", {})
        else:
            vitals = {}

        # Normalize vitals to dictionary for tool consumption
        vitals_obj: Any = vitals
        if hasattr(vitals_obj, "model_dump"):
            vitals_dict = vitals_obj.model_dump()
        elif hasattr(vitals_obj, "dict"):
            vitals_dict = vitals_obj.dict()
        elif isinstance(vitals_obj, dict):
            vitals_dict = vitals_obj
        else:
            vitals_dict = {}

        try:
            # 2. Calculation (deterministic or probabilistic LLM)
            raw_scores = {}
            if self.use_probabilistic:
                results = self._calc_scores_with_llm(vitals_dict)
                simple_report = ""
                for score_name in ["NEWS", "MEWS"]:
                    entry = results.get(score_name, {}) if isinstance(results, dict) else {}
                    sc = entry.get("score")
                    risk = entry.get("risk_level", "Unknown")
                    missing = entry.get("missing_fields", [])
                    assumptions = entry.get("assumptions", [])
                    simple_report += f"{score_name}: score={sc}, risk={risk}, missing={missing}, assumptions={assumptions}\n"
                    if sc is not None:
                        try:
                            raw_scores[score_name] = float(sc)
                        except Exception:
                            raw_scores[score_name] = sc
            else:
                results = {}
                simple_report = ""
                for score in ["NEWS", "MEWS"]:
                    try:
                        logger.debug("Calculating scores via tool...")
                        res_text = calculate_clinical_score(vitals_dict, score)
                        results[score] = res_text
                        simple_report += f"{score}: {res_text}\n"
                        # best-effort numeric extraction
                        try:
                            prefix = f"SCORE TOTAL {score}: "
                            if prefix in res_text:
                                num_part = res_text.split(prefix,1)[1].split("\n",1)[0]
                                raw_scores[score] = float(num_part.strip())
                        except Exception:
                            pass
                        logger.info(f"🧮 {score} Score Calculated: {results[score]}")
                    except Exception as e:
                        logger.error(f"Calculation Error {score}: {e}")
                        results[score] = f"Error calculating {score}: {str(e)}"

            # 3. Model Invocation for Interpretation (NLU)
            vitals_json = json.dumps(vitals_dict, default=str)
            calc_payload = results if self.use_probabilistic else json.dumps(results, indent=2)
            context_msg = f"""
            [PRE-CALCULATED SCORES]
            Analyze the following calculation outputs carefully. Note any [ESTIMATED] tags.
            
            Input Vitals: {vitals_json}
            
            Calculation Output:
            {calc_payload}
            
            [TASK]
            Analyze the calculated scores above.
            1. If the score status is [ESTIMATED], mention explicitly which data was missing in your analysis.
            2. Fill 'analyzed_scores' with the numeric values.
            3. Synthesize the 'overall_risk_assessment'.
            """

            messages = [
                SystemMessage(content=MATHEMATICIAN_SYSTEM_PROMPT),
                HumanMessage(content=context_msg)
            ]
            
            # 4. Risk analysis by LLM
            logger.debug("⏳ Calling LLM for Risk Analysis...")
            response = run_with_timeout(self.model.invoke, messages, timeout=180, retries=2)

            if hasattr(response, "model_dump"):
                result_data = response.model_dump()
            elif hasattr(response, "dict"):
                result_data = response.dict()
            else:
                result_data = response
                
            result_data["calculated_raw"] = raw_scores or results

            logging.info(f"✅ Mathematician Complete: {simple_report}")
            
            return {
                "extracted_data": extracted_data,
                "risk_score_report": simple_report,
                "risk_analysis": result_data 
            }

        except Exception as e:
            logger.error(f"❌ Mathematician Critical Error: {e}")
            traceback.print_exc()
            return {"risk_score_report": f"Critical Error in Mathematician: {str(e)}"}