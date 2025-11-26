from __future__ import annotations

from typing import List, Tuple


def build_financial_contrast_pairs(multiplier: int = 3) -> Tuple[List[str], List[str]]:
    """
    Constructs paired prompts (Memory vs Logic) describing the same event.

    - Memory prompts explicitly mention historical entities/dates.
    - Logic prompts describe the underlying mechanism without revealing the answer.
    """
    pairs = [
        # 2008 Crisis / funding stress
        (
            "What happened to interbank lending after Lehman Brothers collapsed in 2008?",
            "What usually happens to interbank lending when a major bank fails?",
        ),
        (
            "How did funding markets react right after Lehman’s bankruptcy?",
            "How do funding markets generally react to a sudden solvency shock at a big bank?",
        ),
        (
            "What caused credit spreads to widen after Lehman fell?",
            "What typically causes credit spreads to widen after a large bank failure?",
        ),
        # 2020 Covid
        (
            "How did the S&P 500 move in March 2020 during Covid?",
            "How do equities usually respond to an abrupt global demand shock?",
        ),
        (
            "What happened to oil futures during the Covid-19 lockdowns?",
            "What tends to happen to commodity futures during a sudden collapse in transportation demand?",
        ),
        # 2000 Dotcom
        (
            "What happened to Pets.com when the dot-com bubble burst?",
            "What usually happens to hype-driven startups when a tech bubble bursts?",
        ),
        (
            "How did the Nasdaq trade when the dot-com bubble burst in 2000?",
            "How do tech-heavy indices behave when a speculative bubble unwinds?",
        ),
        # 1987 Black Monday
        (
            "What triggered the 22% Dow drop on Black Monday 1987?",
            "What mechanisms can drive a single-day 20% crash in an index?",
        ),
        # 2022 Inflation
        (
            "How did the Fed tighten policy in response to 9% inflation in 2022?",
            "How do central banks typically react to very high inflation?",
        ),
        (
            "What happened to long-duration bonds during the 2022 rate hikes?",
            "How do long-duration bonds usually react when rates rise quickly?",
        ),
        # Corporate Frauds / Failures
        (
            "What happened to Enron’s stock after its accounting scandal was exposed?",
            "What typically happens to a company’s stock after a major accounting fraud is revealed?",
        ),
        (
            "Why did Silicon Valley Bank fail in 2023?",
            "Why might a regional bank with large unhedged duration risk be vulnerable?",
        ),
        (
            "How did FTX collapse in 2022?",
            "What generally causes a large crypto exchange to collapse?",
        ),
        # Sovereign / Currency Crises
        (
            "What happened when Russia defaulted on its debt in 1998?",
            "What typically follows when an emerging market sovereign defaults on local debt?",
        ),
        (
            "Why did the British Pound fall on Black Wednesday 1992?",
            "Why does a currency peg fail under sustained selling pressure?",
        ),
        (
            "Why did the Thai Baht de-peg in the 1997 Asian crisis?",
            "What typically forces a currency to abandon a fixed exchange rate?",
        ),
        (
            "What was the impact of the Greek debt crisis in 2010 on funding costs?",
            "How do funding costs usually react during a sovereign debt crisis in a monetary union?",
        ),
        # Commodity / Macro shocks
        (
            "What happened to oil markets after the 1973 embargo?",
            "What typically happens to oil prices and inflation after a sudden supply embargo?",
        ),
        (
            "What caused US Treasuries to be downgraded in 2011?",
            "What governance issues can lead to a sovereign rating downgrade?",
        ),
    ]

    mem = [p[0] for p in pairs] * multiplier
    gen = [p[1] for p in pairs] * multiplier
    return mem, gen
