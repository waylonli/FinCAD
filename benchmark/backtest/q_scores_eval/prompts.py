"""Category prompts for risk-factor scoring and CAD prior templates.

The five category prompts are copied verbatim from
``abrdn-risk-factor-eval/src/prompts/risk_factor_prompts.py``.

Two additional prior-prompt templates are defined for context-aware
decoding (CAD):
- ``NO_CONTEXT_PRIOR_TEMPLATE``  -- generic evaluation without company context
- ``BIAS_AMPLIFIED_PRIOR_TEMPLATE`` -- explicitly names the ticker to trigger
  memorised knowledge (adversarial CAD, proposal Section 2.1)
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Category prompts (verbatim from abrdn)
# ---------------------------------------------------------------------------

INDUSTRY_PROMPT = """
Evaluate the following aspects of the mentioned company's industry as described in the input annual report. Based on these considerations, provide a score ranging from 1 to 100 for the overall industry performance and a brief explanation.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.

Here are the aspects to consider:
Is the industry growing?
Are industry prices typically inflationary?
Is the industry highly cyclical or defensive?
What is the typical level of margins in the industry?
To what extent is the industry regulated, and how?
Is new capital entering the market, or is capital leaving?
Is there a risk of disruptive new technology?
What is the level of consolidation or fragmentation among suppliers and customers in the industry?
Is the industry heavily capital-intensive?

The input annual report is as follows:
{input_annual_report}
"""

BUSINESS_MODEL_PROMPT = """
Evaluate the following aspects of the mentioned company's business model as described in the input annual report. Based on these considerations, provide a score ranging from 1 to 100 for the overall business model and a brief explanation.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.

Here are the aspects to consider:
The most fundamental question here is whether the company can sustainably earn a return above its cost of capital. This means assessing its own particular competitive strengths and weaknesses.
Quality increases with the size of the company's 'economic moat' so high barriers to entry that prevent competitors eating in to outsized returns should boost the overall evaluation.
It also matters whether the company's business model allows it the opportunity to grow while earning attractive returns, so assessing the likely growth rate of the business, and considering incremental as well as overall margins and returns, is important.

The input annual report is as follows:
{input_annual_report}
"""

FINANCIAL_STRENGTH_PROMPT = """
Evaluate the following aspects of the mentioned company's financial strength as described in the input annual report. Based on these considerations, provide a score ranging from 1 to 100 for the overall financial strength and a brief explanation.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.

Here are the aspects to consider:
All other things being equal, a business that consistently translates its profits into cash is of higher Quality than one that fails to do that.
A strong balance sheet can be a key aspect of its Quality, both as downside protection if and when the company hits a bump in the road, and as a source of capital for further growth.
A view of how the company manages its financial risks, such as exposures to commodity prices, foreign currencies or changes in interest rates, should also feed into the Financials score.

The input annual report is as follows:
{input_annual_report}
"""

MANAGEMENT_PROMPT = """
Evaluate the following aspects of the mentioned company's management team as described in the input annual report. Based on these considerations, provide a score ranging from 1 to 100 for the overall management quality and a brief explanation.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.

Here are the aspects to consider:
Can we see evidence that management has made good capital allocation decisions in the past?
Do they have a clear strategic vision, and are they executing against it?
How well does management communicate with stakeholders?
Do incentives appear to be aligned with long-term value creation?
Are there signs of effective risk management practices?
We are not only interested in the quality of the company's CEO, but also the broader executive management team around them.

The input annual report is as follows:
{input_annual_report}
"""

ESG_PROMPT = """
Evaluate the following aspects of the mentioned company's ESG performance as described in the input annual report. Based on these considerations, provide a score ranging from 1 to 100 for the overall ESG performance and a brief explanation.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.

Here are the aspects to consider:
What is your view on the quality of the governance and oversight of the business and management?
What are the most material environmental, social, and operational governance issues the company must manage and how are these being addressed?
What is your conclusion on how the management of the most material ESG risks and opportunities will impact the business' operational performance and valuation?

The input annual report is as follows:
{input_annual_report}
"""

DEFAULT_CATEGORY_PROMPTS: Dict[str, str] = {
    "Industry": INDUSTRY_PROMPT,
    "Business Model": BUSINESS_MODEL_PROMPT,
    "Financial Strength": FINANCIAL_STRENGTH_PROMPT,
    "Management": MANAGEMENT_PROMPT,
    "ESG": ESG_PROMPT,
}

# ---------------------------------------------------------------------------
# CAD prior-prompt templates
# ---------------------------------------------------------------------------

NO_CONTEXT_PRIOR_TEMPLATE = """
Evaluate the following aspect of a company's {category}.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.
"""

BIAS_AMPLIFIED_PRIOR_TEMPLATE = """
Using your internal knowledge about {symbol}, evaluate the {category} of {symbol}.
Return your analysis as a JSON object with exactly these keys:
- "score": integer between 0 and 100
- "explanation": string with a concise rationale
Respond with valid JSON only—do not include any additional text before or after the JSON object.
"""


def build_prior_prompt(
    category: str,
    symbol: str,
    mode: str,
    neg_prompt_builder=None,
) -> str:
    """Build a prior prompt for CAD decoding.

    Parameters
    ----------
    category:
        One of the five scoring categories (e.g. "Industry").
    symbol:
        Ticker symbol (e.g. "AAPL").
    mode:
        ``"no_context"`` -- generic evaluation without company-specific info.
        ``"bias_amplified"`` -- names the ticker to trigger memorised knowledge.
        ``"optimized"`` -- use DSPy-optimized instruction via *neg_prompt_builder*.
    neg_prompt_builder:
        ``NegativePromptBuilder`` instance (required when mode is ``"optimized"``).
    """
    if mode == "optimized" and neg_prompt_builder is not None:
        # Extract the full task instruction from the category prompt,
        # stripping the filing text placeholder so both x_ctx and x_prior
        # share the same task framing and output format.
        full_prompt = DEFAULT_CATEGORY_PROMPTS.get(category, "")
        task_instruction = full_prompt.split("\nThe input annual report is as follows:")[0].strip()
        return neg_prompt_builder.build(
            entity=symbol, task_prompt=task_instruction,
        )
    if mode == "no_context":
        return NO_CONTEXT_PRIOR_TEMPLATE.format(category=category)
    if mode == "bias_amplified":
        return BIAS_AMPLIFIED_PRIOR_TEMPLATE.format(symbol=symbol, category=category)
    raise ValueError(f"Unknown prior mode: {mode!r}. Use 'no_context', 'bias_amplified', or 'optimized'.")
