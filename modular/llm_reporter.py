"""
LLM Integration for Safety Report Generation using Ollama
"""
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
)


@dataclass
class SafetyReport:
    """Generated safety report from LLM."""
    summary: str
    risk_assessment: str
    recommendations: List[str]
    business_impact: str
    action_items: List[str]
    raw_response: str


class LLMReporter:
    """
    LLM-based safety report generator using local Ollama.
    """
    
    def __init__(self, 
                 base_url: str = OLLAMA_BASE_URL,
                 model: str = OLLAMA_MODEL,
                 timeout: int = OLLAMA_TIMEOUT):
        """
        Initialize LLM reporter.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                return self.model.split(':')[0] in model_names or any(
                    self.model in m.get('name', '') for m in models
                )
            return False
        except Exception:
            return False
    
    def _build_prompt(self, 
                      violation_counts: Dict[str, int],
                      total_frames: int,
                      duration_seconds: float,
                      final_risk_score: int,
                      context: str = "construction site") -> str:
        """Build the prompt for the LLM."""
        
        total_violations = sum(violation_counts.values())
        violations_per_minute = (total_violations / duration_seconds * 60) if duration_seconds > 0 else 0
        
        violations_detail = "\n".join([
            f"  - {v_type}: {count} occurrences"
            for v_type, count in violation_counts.items()
        ]) if violation_counts else "  - No violations detected"
        
        prompt = f"""You are an expert workplace safety consultant analyzing AI-detected safety violations from a {context} monitoring system.

## Analysis Data

**Monitoring Session:**
- Duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)
- Frames Analyzed: {total_frames}
- Final Risk Score: {final_risk_score}/1000

**Violations Detected:**
{violations_detail}

**Violation Rate:** {violations_per_minute:.1f} violations per minute

## Your Task

Please provide a comprehensive safety report with the following sections:

### 1. EXECUTIVE SUMMARY
Brief overview of the safety situation (2-3 sentences).

### 2. RISK ASSESSMENT
- Severity level (LOW/MEDIUM/HIGH/CRITICAL)
- Key risk factors identified
- Potential consequences if not addressed

### 3. SPECIFIC RECOMMENDATIONS
Actionable steps to address each type of violation:
{chr(10).join([f'- For {v}: specific recommendation' for v in violation_counts.keys()]) if violation_counts else '- General safety maintenance recommendations'}

### 4. BUSINESS IMPACT
- Potential costs of non-compliance (fines, injuries, downtime)
- ROI of implementing recommendations
- Insurance and liability considerations

### 5. ACTION ITEMS
Priority-ordered list of immediate actions (with responsible party suggestions):
1. [HIGH PRIORITY] ...
2. [MEDIUM PRIORITY] ...
3. [LOW PRIORITY] ...

### 6. COMPLIANCE NOTES
Relevant OSHA regulations and standards that apply.

Please be specific, practical, and business-focused in your recommendations."""

        return prompt
    
    def generate_report(self,
                       violation_counts: Dict[str, int],
                       total_frames: int,
                       duration_seconds: float,
                       final_risk_score: int,
                       context: str = "construction site") -> SafetyReport:
        """
        Generate a safety report using the LLM.
        
        Args:
            violation_counts: Dictionary of violation type -> count
            total_frames: Total frames analyzed
            duration_seconds: Duration of analysis
            final_risk_score: Final cumulative risk score
            context: Context description (e.g., "construction site", "warehouse")
            
        Returns:
            SafetyReport with generated content
        """
        prompt = self._build_prompt(
            violation_counts, total_frames, duration_seconds, 
            final_risk_score, context
        )
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000,
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            raw_response = result.get('response', '')
            
            # Parse the response into sections
            return self._parse_response(raw_response)
            
        except requests.exceptions.Timeout:
            return SafetyReport(
                summary="Report generation timed out. Please try again.",
                risk_assessment="Unable to assess - timeout occurred.",
                recommendations=["Retry report generation with shorter analysis duration."],
                business_impact="Unable to calculate - timeout occurred.",
                action_items=["Ensure Ollama is running properly."],
                raw_response="TIMEOUT ERROR"
            )
        except Exception as e:
            return SafetyReport(
                summary=f"Report generation failed: {str(e)}",
                risk_assessment="Unable to assess due to error.",
                recommendations=["Check Ollama connection and try again."],
                business_impact="Unable to calculate due to error.",
                action_items=["Verify Ollama is running: `ollama serve`"],
                raw_response=f"ERROR: {str(e)}"
            )
    
    def _parse_response(self, raw_response: str) -> SafetyReport:
        """Parse LLM response into structured sections."""
        
        # Default values
        summary = ""
        risk_assessment = ""
        recommendations = []
        business_impact = ""
        action_items = []
        
        sections = raw_response.split("###")
        
        for section in sections:
            section_lower = section.lower().strip()
            content = section.strip()
            
            if "executive summary" in section_lower or "summary" in section_lower[:30]:
                summary = self._extract_content(content)
            elif "risk assessment" in section_lower:
                risk_assessment = self._extract_content(content)
            elif "specific recommendation" in section_lower or "recommendations" in section_lower[:30]:
                recommendations = self._extract_list(content)
            elif "business impact" in section_lower:
                business_impact = self._extract_content(content)
            elif "action items" in section_lower or "action item" in section_lower[:30]:
                action_items = self._extract_list(content)
        
        # Fallback if parsing failed
        if not summary and raw_response:
            summary = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
        
        return SafetyReport(
            summary=summary or "No summary available.",
            risk_assessment=risk_assessment or "Risk assessment not available.",
            recommendations=recommendations or ["Review raw report for recommendations."],
            business_impact=business_impact or "Business impact analysis not available.",
            action_items=action_items or ["Review raw report for action items."],
            raw_response=raw_response
        )
    
    def _extract_content(self, section: str) -> str:
        """Extract content from a section, removing the header."""
        lines = section.split('\n')
        # Skip the first line (header) and join the rest
        content_lines = [l.strip() for l in lines[1:] if l.strip()]
        return '\n'.join(content_lines)
    
    def _extract_list(self, section: str) -> List[str]:
        """Extract list items from a section."""
        items = []
        lines = section.split('\n')
        
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        (len(line) > 2 and line[0].isdigit() and line[1] in '.)')):
                # Remove bullet/number prefix
                if line.startswith('-') or line.startswith('•'):
                    item = line[1:].strip()
                else:
                    item = line[2:].strip() if line[1] in '.)' else line[3:].strip()
                if item:
                    items.append(item)
        
        return items if items else [section.strip()]


def generate_quick_summary(violation_counts: Dict[str, int], 
                          risk_score: int) -> str:
    """
    Generate a quick text summary without LLM.
    
    Args:
        violation_counts: Dictionary of violations
        risk_score: Current risk score
        
    Returns:
        Quick summary string
    """
    if not violation_counts:
        return "✅ No safety violations detected. All personnel appear to be in compliance with PPE requirements."
    
    total = sum(violation_counts.values())
    severity = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 60 else "HIGH" if risk_score < 100 else "CRITICAL"
    
    summary_parts = [f"⚠️ **{severity} RISK** - Detected {total} violation(s):"]
    
    for v_type, count in violation_counts.items():
        if v_type == "NO_HELMET":
            summary_parts.append(f"- 🪖 **No Helmet**: {count} instance(s) - Head injury risk")
        elif v_type == "NO_VEST":
            summary_parts.append(f"- 🦺 **No Safety Vest**: {count} instance(s) - Visibility hazard")
        elif v_type == "RUNNING":
            summary_parts.append(f"- 🏃 **Running Detected**: {count} instance(s) - Trip/fall risk")
        elif v_type == "FORKLIFT_RISK":
            summary_parts.append(f"- 🚜 **Forklift Proximity**: {count} instance(s) - Collision hazard")
        elif v_type == "BAD_LIFT":
            summary_parts.append(f"- 💪 **Bad Lifting Posture**: {count} instance(s) - Back injury risk")
        elif v_type == "SLIP_RISK":
            summary_parts.append(f"- ⚡ **Slip Risk**: {count} instance(s) - Fall hazard")
        else:
            summary_parts.append(f"- ⚠️ **{v_type}**: {count} instance(s)")
    
    return '\n'.join(summary_parts)
