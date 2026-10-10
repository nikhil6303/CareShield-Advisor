"""
ollama_service.py
=================
Local Open-Weight AI Explanation Service using Qwen3 1.7B through Ollama.

Architecture & Safety Principles:
---------------------------------
1. Independent from XGBoost:
   XGBoost is the SOLE churn prediction algorithm. Qwen3 never calculates
   churn probabilities, never overrides model scores, and never acts as a predictor.

2. Evidence-Based Decision Support:
   Prompts pass structured factual evidence ONLY (XGBoost probability, risk tier,
   SHAP drivers, and observable plan records). Strictly forbids hallucinated customer
   history, correlation-causation confusion, and medical diagnoses.

3. Low Resource Overhead (8 GB RAM / Windows Friendly):
   - Targets Qwen3 1.7B (`qwen3:1.7b`) by default.
   - Concurrency lock limits in-flight local inference to 1 thread at a time.
   - Configurable token limit (default 450 tokens) prevents runaway generations.
   - In-memory cache avoids duplicate Ollama inference for the same member.
   - Short, structured prompts keep KV-cache compact.

4. Non-Blocking & Resilient:
   If Ollama is stopped, offline, or times out, the core XGBoost prediction
   and rule engine continue without interruption.

Author  : CTS NPN Project / CareShield Advisor
Created : 2026-10-10
"""

import os
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Configuration via environment variables
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "true").strip().lower() not in ("false", "0", "no")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "30"))  # seconds
MAX_OUTPUT_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "450"))

# Concurrency lock to prevent multiple simultaneous LLM calls from overwhelming 8 GB RAM
_inference_lock = threading.Lock()

# In-memory explanation cache (member_id -> explanation dict)
_explanation_cache: Dict[str, Dict[str, Any]] = {}


def get_ollama_config() -> Dict[str, Any]:
    """Return current configuration for the Ollama explanation service."""
    return {
        "enabled": OLLAMA_ENABLED,
        "host": OLLAMA_HOST,
        "model": OLLAMA_MODEL,
        "timeout_seconds": OLLAMA_TIMEOUT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def check_ollama_health() -> Dict[str, Any]:
    """
    Check if the local Ollama instance is reachable and whether the target model is loaded.
    Timeout is intentionally kept low (2s) so health checks are instantaneous.
    Returns a disabled status immediately when OLLAMA_ENABLED=false.
    """
    if not OLLAMA_ENABLED:
        return {
            "available": False,
            "status": "disabled",
            "host": OLLAMA_HOST,
            "target_model": OLLAMA_MODEL,
            "model_installed": False,
            "installed_models": [],
            "error": "AI explanation service is disabled (OLLAMA_ENABLED=false).",
            "instructions": "Set OLLAMA_ENABLED=true and configure OLLAMA_HOST to enable AI explanations.",
        }
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match model name or prefix (e.g. 'qwen3:1.7b' or 'qwen3:1.7b-instruct')
            target_installed = any(
                OLLAMA_MODEL.lower() in m.lower() or m.lower().startswith(OLLAMA_MODEL.lower().split(":")[0])
                for m in models
            )
            return {
                "available": True,
                "status": "online",
                "host": OLLAMA_HOST,
                "target_model": OLLAMA_MODEL,
                "model_installed": target_installed,
                "installed_models": models,
                "instructions": None if target_installed else f"Run 'ollama pull {OLLAMA_MODEL}' in Windows CMD to download the model."
            }
        return {
            "available": False,
            "status": "unhealthy",
            "host": OLLAMA_HOST,
            "target_model": OLLAMA_MODEL,
            "model_installed": False,
            "installed_models": [],
            "error": f"Ollama HTTP {resp.status_code}",
            "instructions": "Ensure Ollama is running at http://localhost:11434."
        }
    except requests.exceptions.RequestException as exc:
        return {
            "available": False,
            "status": "offline",
            "host": OLLAMA_HOST,
            "target_model": OLLAMA_MODEL,
            "model_installed": False,
            "installed_models": [],
            "error": str(exc),
            "instructions": f"Start Ollama desktop or run 'ollama run {OLLAMA_MODEL}' in Windows CMD."
        }



def _build_evidence_prompt(
    industry_name: str,
    churn_probability: float,
    risk_level: str,
    shap_drivers: List[Dict[str, Any]],
    member_info: Dict[str, Any],
) -> str:
    """
    Construct a compact, evidence-based prompt containing ONLY structured signals.
    """
    # Format top 4 drivers compactly
    driver_lines = []
    for d in shap_drivers[:4]:
        feat = d.get("label", d.get("feature", "Unknown"))
        val = d.get("value", "N/A")
        direction = "increases churn risk" if d.get("direction") == "increases_risk" else "reduces churn risk"
        sv = d.get("shap_value", 0.0)
        driver_lines.append(f"- {feat}: {val} ({direction}, SHAP impact {sv:+.3f})")

    drivers_text = "\n".join(driver_lines) if driver_lines else "None recorded"

    # Relevant observable plan facts
    tenure = member_info.get("Tenure_Months", "N/A")
    plan = member_info.get("Insurance_Type", "N/A")
    cost = member_info.get("Avg_Out_Of_Pocket_Cost", "N/A")
    billing = "Yes" if member_info.get("Billing_Issues") == 1 else "No"
    portal = "Active" if member_info.get("Portal_Usage") == 1 else "Inactive"
    days_since = member_info.get("Days_Since_Last_Visit", "N/A")
    sat = member_info.get("Overall_Satisfaction", "N/A")

    prompt = f"""You are a healthcare retention analyst assistant for CareShield.
Analyze the following empirical evidence for a health plan member.

[FACTUAL EVIDENCE]
- Industry: {industry_name}
- XGBoost Predicted Churn Risk: {churn_probability * 100:.1f}% ({risk_level} Risk Category)
- Membership Tenure: {tenure} months
- Coverage Plan: {plan}
- Out-of-Pocket Spend: ${cost}
- Billing Issue Recorded: {billing}
- Patient Portal Usage: {portal}
- Days Since Last Clinical Visit: {days_since} days
- Satisfaction Rating: {sat} / 5.0

[MEASURED SHAP RISK DRIVERS]
{drivers_text}

[CONSTRAINTS]
- Do NOT alter the churn probability or calculate a new score.
- Do NOT assume correlation implies causation.
- Do NOT invent unrecorded medical diagnoses or emotional states.
- Keep your response structured, concise, and professional.

Return a JSON object with EXACTLY these four keys:
{{
  "summary_explanation": "Concise 2-sentence explanation of why the XGBoost model flagged this member as {risk_level} risk.",
  "contributing_factors": ["Point 1 based on evidence", "Point 2 based on evidence", "Point 3 based on evidence"],
  "retention_actions": ["Actionable intervention 1", "Actionable intervention 2"],
  "limitations": "Brief statement of unmeasured variables and that coordinator review is required."
}}
JSON Response:"""
    return prompt


def _parse_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Safely parse the LLM text output into structured sections, handling potential markdown code blocks.
    """
    clean_text = raw_text.strip()

    # Strip markdown ```json ... ``` blocks if present
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_text:
        clean_text = clean_text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict) and "summary_explanation" in parsed:
            return {
                "summary_explanation": str(parsed.get("summary_explanation", "")),
                "contributing_factors": parsed.get("contributing_factors", []),
                "retention_actions": parsed.get("retention_actions", []),
                "limitations": str(parsed.get("limitations", "Requires human clinical review; observational signals only."))
            }
    except Exception:
        pass

    # Fallback parser if LLM produced plain text instead of valid JSON
    return {
        "summary_explanation": raw_text[:300].strip(),
        "contributing_factors": ["Empirical risk signals derived from XGBoost feature attribution."],
        "retention_actions": ["Review member outreach records and schedule coordinator follow-up."],
        "limitations": "Analysis based on observational dataset features. Human review required before outreach."
    }


def generate_ai_explanation(
    member_id: Optional[str],
    churn_probability: float,
    risk_level: str,
    shap_drivers: List[Dict[str, Any]],
    member_info: Dict[str, Any],
    industry_name: str = "Healthcare (CareShield Health Plan)",
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Generate an evidence-based explanation using local Qwen3 1.7B through Ollama.
    Guarantees that XGBoost predictions are never blocked or altered.
    """
    # 1. Bail out immediately when AI service is disabled
    if not OLLAMA_ENABLED:
        return {
            "success": False,
            "error": "AI explanation service is disabled on this deployment.",
            "details": "Set OLLAMA_ENABLED=true and configure a reachable OLLAMA_HOST to enable.",
            "instructions": "Configure OLLAMA_ENABLED and OLLAMA_HOST environment variables.",
            "is_ai_generated": False,
            "model": OLLAMA_MODEL,
        }

    # 2. Check in-memory cache if member_id is provided
    if member_id and not bypass_cache and member_id in _explanation_cache:
        cached = dict(_explanation_cache[member_id])
        cached["from_cache"] = True
        return cached

    # 3. Check Ollama availability before acquiring thread lock
    health = check_ollama_health()
    if not health.get("available"):
        return {
            "success": False,
            "error": "Local Ollama service is not currently running.",
            "details": health.get("error", "Connection refused at " + OLLAMA_HOST),
            "instructions": f"Start Ollama and run 'ollama run {OLLAMA_MODEL}' in Windows CMD.",
            "is_ai_generated": False,
            "model": OLLAMA_MODEL,
        }

    # 3. Construct prompt
    prompt = _build_evidence_prompt(
        industry_name=industry_name,
        churn_probability=churn_probability,
        risk_level=risk_level,
        shap_drivers=shap_drivers,
        member_info=member_info,
    )

    # 4. Acquire lock to prevent concurrent inferences on 8 GB RAM systems
    acquired = _inference_lock.acquire(timeout=10.0)
    if not acquired:
        return {
            "success": False,
            "error": "Local AI explanation engine is busy processing another request.",
            "instructions": "Please wait a moment and try again. Concurrent local inference is throttled to preserve memory.",
            "is_ai_generated": False,
            "model": OLLAMA_MODEL,
        }

    start_time = time.perf_counter()
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": MAX_OUTPUT_TOKENS,
                "temperature": 0.2,   # Low temperature for factual consistency
                "top_p": 0.9,
            }
        }

        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        elapsed_sec = round(time.perf_counter() - start_time, 2)

        if resp.status_code != 200:
            logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
            return {
                "success": False,
                "error": f"Ollama generation failed with status code {resp.status_code}.",
                "details": resp.text[:200],
                "instructions": f"Verify model '{OLLAMA_MODEL}' is pulled and healthy via 'ollama list'.",
                "is_ai_generated": False,
                "model": OLLAMA_MODEL,
            }

        response_json = resp.json()
        raw_output = response_json.get("response", "")
        if not raw_output.strip():
            return {
                "success": False,
                "error": "Ollama returned an empty response.",
                "is_ai_generated": False,
                "model": OLLAMA_MODEL,
            }

        parsed = _parse_llm_response(raw_output)

        result = {
            "success": True,
            "is_ai_generated": True,
            "model": OLLAMA_MODEL,
            "latency_seconds": elapsed_sec,
            "explanation": {
                "summary": parsed["summary_explanation"],
                "contributing_factors": parsed["contributing_factors"],
                "recommended_actions": parsed["retention_actions"],
                "limitations": parsed["limitations"],
            },
            "disclaimer": "AI-generated strategic synthesis powered by local Qwen3 1.7B. Decision support only; requires human clinical and care coordinator review.",
            "from_cache": False,
        }

        # Cache if valid member_id
        if member_id:
            _explanation_cache[member_id] = result

        return result

    except requests.exceptions.Timeout:
        logger.warning(f"Ollama request timed out after {OLLAMA_TIMEOUT}s.")
        return {
            "success": False,
            "error": f"AI generation timed out after {OLLAMA_TIMEOUT} seconds.",
            "instructions": "Local CPU/RAM may be under load. Try closing heavy background applications.",
            "is_ai_generated": False,
            "model": OLLAMA_MODEL,
        }
    except Exception as exc:
        logger.error(f"Unexpected error in Ollama explanation service: {str(exc)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to communicate with local AI service: {str(exc)}",
            "is_ai_generated": False,
            "model": OLLAMA_MODEL,
        }
    finally:
        _inference_lock.release()


def clear_explanation_cache() -> None:
    """Clear in-memory cached AI explanations."""
    global _explanation_cache
    _explanation_cache.clear()
