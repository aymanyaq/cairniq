"""
DSPy signatures for CairnIQ.
Define the input/output specifications for each agent task.
"""
from agent.dspy_setup import dspy


class PortfolioAnalysis(dspy.Signature):
    """
    Analyze portfolio metrics, risk factors, and diversification.
    Identify strengths, weaknesses, and key risks based on the data.
    STRICT RULE: NEVER invent or hallucinate data, metrics, or holdings not present in the input.
    """
    portfolio_data: str = dspy.InputField(desc="Holdings, values, and account info")
    risk_metrics: str = dspy.InputField(desc="Sharpe, Beta, VaR, correlations")
    analysis_summary: str = dspy.OutputField(desc="Bullet points of key findings (Performance, Risk, Concentration, Fees)")
    risk_flags: str = dspy.OutputField(desc="Specific risks identified (e.g. 'High Tech exposure', 'Low liquidity')")

class PortfolioStrategy(dspy.Signature):
    """
    Formulate actionable investment advice based on analysis and user context.
    STRICT RULE: NEVER invent or hallucinate data, metrics, or holdings not present in the input.
    """
    analysis_summary: str = dspy.InputField(desc="Key findings from portfolio analysis")
    risk_flags: str = dspy.InputField(desc="Identified risks to address")
    user_context: str = dspy.InputField(desc="User profile, goals, and questions")
    advice: str = dspy.OutputField(
        desc="Actionable recommendations. Use tables. Format money as '$10,000'. No backslashes in currency."
    )


class MarketAnalysis(dspy.Signature):
    """
    Analyze a stock or ETF and provide a buy/sell/hold recommendation.
    Use fundamentals, technicals, and news to form a thesis.
    STRICT RULE: NEVER invent or hallucinate specific numbers (e.g. EBIT, guidance, dates). If data is unavailable in the context, state 'Data Unavailable'.
    If the verdict is HOLD/WAIT/NEUTRAL/AVOID, do not provide a buy-entry or half-position workaround.
    """
    symbol: str = dspy.InputField(desc="Stock or ETF ticker symbol")
    fundamentals: str = dspy.InputField(
        desc="Fundamental data: PE ratio, dividends, 52-week range, market cap"
    )
    technicals: str = dspy.InputField(
        desc="Technical indicators: RSI, MACD, moving averages, support/resistance"
    )
    news: str = dspy.InputField(
        desc="Recent news and sentiment about the stock"
    )
    summary: str = dspy.OutputField(
        desc="High-level summary of the situation in plain English. Use analogies where helpful (e.g. 'like a car driving up a hill')."
    )
    analysis: str = dspy.OutputField(
        desc="Final synthesized verdict and recommendation. Explain technical terms and WHY conflicts exist."
    )
    key_conflicts: str = dspy.OutputField(
        desc="Identify any contradictions in the data (e.g. Analysts say Hold but Technicals say Buy)"
    )
    risk_factors: str = dspy.OutputField(
        desc="Specific downside risks that justify caution, including missing foundation evidence, high beta, short-interest mechanics, or portfolio concentration when present"
    )
    bull_case: str = dspy.OutputField(
        desc="Top 3 bullish arguments (Pros) supporting a buy thesis"
    )
    bear_case: str = dspy.OutputField(
        desc="Top 3 bearish arguments (Cons) supporting a sell/avoid thesis"
    )
    catalysts: str = dspy.OutputField(
        desc="Recent positive or negative catalysts (news, earnings, macro) driving the stock"
    )


class RiskAssessment(dspy.Signature):
    """
    Assess the risk level of a proposed investment action.
    Check for position sizing, diversification, and common pitfalls.
    STRICT RULE: NEVER invent or hallucinate data, metrics, or holdings not present in the input.
    """
    portfolio_context: str = dspy.InputField(
        desc="Current portfolio holdings and total value"
    )
    proposed_action: str = dspy.InputField(
        desc="The investment recommendation being evaluated"
    )
    user_profile: str = dspy.InputField(
        desc="User's risk tolerance, age, and investment horizon"
    )
    assessment: str = dspy.OutputField(
        desc="Risk assessment: potential risks, position sizing advice, and any warnings. Be concise."
    )


class NewsAnalysis(dspy.Signature):
    """
    Analyze news and extract investment-relevant insights.
    STRICT RULE: NEVER invent or hallucinate news events, dates, or data not present in the articles.
    """
    news_articles: str = dspy.InputField(
        desc="Collection of news articles and headlines"
    )
    focus_area: str = dspy.InputField(
        desc="Stocks, sectors, or themes to focus on"
    )
    summary: str = dspy.OutputField(
        desc="Key takeaways: what's happening, market sentiment, and investment implications"
    )

class ContextExtraction(dspy.Signature):
    """
    Extract user profile details, key facts, and relationship triplets for a knowledge graph.
    STRICT RULE: NEVER invent or hallucinate facts or relationships not explicitly stated by the user.
    """
    user_message: str = dspy.InputField(
        desc="The raw message from the user."
    )
    profile_updates: str = dspy.OutputField(
        desc="JSON object with updates to user profile (age, income, risk_tolerance, etc) or empty {}."
    )
    new_facts: str = dspy.OutputField(
        desc=(
            "JSON list of durable user facts, preferences, constraints, goals, or account details "
            "explicitly stated by the user. Return [] for casual market-analysis requests. "
            "Do NOT extract facts merely stating the user owns/holds a specific ticker or its "
            "quantity — portfolio holdings are already tracked automatically from the live "
            "portfolio, not from chat messages."
        )
    )
    new_relationships: str = dspy.OutputField(
        desc="List of triplets [source, relation, target] about the USER's identity or explicitly stated interests ONLY. "
             "Valid relations: WORKS_AT, HAS_ACCOUNT, HAS_ACCOUNT_AT, PREFERS, LIVES_IN, "
             "HAS_RISK_TOLERANCE, INTERESTED_IN, MONITORS, TRACKING. "
             "For INTERESTED_IN or MONITORS: only extract if the user EXPLICITLY states interest "
             "(e.g. 'I want to track...', 'keep an eye on...', 'I'm interested in...'). "
             "Do NOT extract interests from casual analysis requests like 'analyze PLTR' or 'what's TSLA at?'. "
             "Do NOT extract relationships about tickers, market events, or third-party entities. "
             "E.g. [['User', 'INTERESTED_IN', 'Quantum Computing'], ['User', 'LIVES_IN', 'Canada']]."
    )


class ThesisGeneration(dspy.Signature):
    """
    Tree-of-Thought: Generate multiple investment theses for comparison.
    Each thesis represents a different perspective on the investment.
    STRICT RULE: NEVER invent or hallucinate specific numbers (e.g. EBIT, guidance, dates). If data is unavailable in the context, state 'Data Unavailable'.
    """
    symbol: str = dspy.InputField(desc="Stock or ETF ticker symbol")
    context: str = dspy.InputField(
        desc="Combined market data: fundamentals, technicals, news, macro context"
    )
    thesis_bull: str = dspy.OutputField(
        desc="BULLISH thesis: Why this is a strong BUY. Include specific catalysts, price targets, and timeframe."
    )
    thesis_bear: str = dspy.OutputField(
        desc="BEARISH thesis: Why this should be SOLD or AVOIDED. Include specific risks, downside targets, and warning signs."
    )
    thesis_neutral: str = dspy.OutputField(
        desc="NEUTRAL thesis: Why HOLD or WAIT is best. Include conditions that would change the recommendation."
    )

class BullThesisGeneration(dspy.Signature):
    """
    Generate a high-fidelity BULLISH investment thesis.
    ANTI-HALLUCINATION RULE: Use ONLY data present in the context.
    NEVER invent, estimate, or hallucinate specific financial or macro metrics (e.g. short interest, M2 liquidity, sector returns, profit margins, analyst targets, or specific dates/EBITDA).
    [cachePoint]
    """
    symbol: str = dspy.InputField(desc="Stock ticker")
    context: str = dspy.InputField(desc="Market data and catalysts")
    thesis: str = dspy.OutputField(desc="Compelling BULLISH thesis with catalysts and targets. Do not rely on pullback labels, social hype, or short interest as bullish without evidence.")
    identified_edge: str = dspy.OutputField(desc="State the specific edge: Time Horizon, Analytical, or Behavioral. If none, label as 'Consensus Trade'.")

class BearThesisGeneration(dspy.Signature):
    """
    Generate a high-fidelity BEARISH investment thesis.
    ANTI-HALLUCINATION RULE: Use ONLY data present in the context.
    NEVER invent, estimate, or hallucinate specific financial or macro metrics (e.g. short interest, M2 liquidity, sector returns, profit margins, analyst targets, or specific dates/EBITDA).
    [cachePoint]
    """
    symbol: str = dspy.InputField(desc="Stock ticker")
    context: str = dspy.InputField(desc="Market data and risks")
    thesis: str = dspy.OutputField(desc="Compelling BEARISH thesis with risks, downside targets, concentration/correlation issues, and invalidated bullish mechanics when relevant.")
    identified_edge: str = dspy.OutputField(desc="State the specific edge: Time Horizon, Analytical, or Behavioral. If none, label as 'Consensus Trade'.")

class NeutralThesisGeneration(dspy.Signature):
    """
    Generate a high-fidelity NEUTRAL/HOLD investment thesis.
    ANTI-HALLUCINATION RULE: Use ONLY data present in the context.
    NEVER invent, estimate, or hallucinate specific financial or macro metrics (e.g. short interest, M2 liquidity, sector returns, profit margins, analyst targets, or specific dates/EBITDA).
    [cachePoint]
    """
    symbol: str = dspy.InputField(desc="Stock ticker")
    context: str = dspy.InputField(desc="Market data and wait conditions")
    thesis: str = dspy.OutputField(desc="Compelling NEUTRAL thesis with wait-and-see conditions. Do not include an entry plan unless conditions for a future bullish upgrade are met.")


class ThesisEvaluation(dspy.Signature):
    """
    Evaluate multiple theses and select the strongest one with confidence scoring.

    JUDGE THE EVIDENCE, NOT THE PROSE. The theses were all written to be
    persuasive. Select the one best SUPPORTED by `evidence_context` — not the one
    that argues most fluently. A claim that appears in a thesis but not in
    `evidence_context` is unsupported and counts AGAINST that thesis.

    ANTI-HALLUCINATION RULE: Use ONLY data present in `evidence_context`.
    NEVER invent, estimate, or hallucinate specific financial or macro metrics (e.g. short interest, M2 liquidity, sector returns, profit margins, analyst targets, or specific dates/EBITDA).
    [cachePoint]
    """
    symbol: str = dspy.InputField(desc="Stock or ETF ticker symbol")
    evidence_context: str = dspy.InputField(
        desc="The underlying evidence the theses were generated from (fundamentals, technicals, news, macro). This is the ONLY admissible source of facts. If a section is empty or missing, treat that evidence as unavailable — do not fill the gap."
    )
    thesis_bull: str = dspy.InputField(desc="The bullish investment thesis")
    thesis_bear: str = dspy.InputField(desc="The bearish investment thesis")
    thesis_neutral: str = dspy.InputField(desc="The neutral investment thesis")

    selected_thesis: str = dspy.OutputField(
        desc="Which thesis is strongest: 'BULL', 'BEAR', or 'NEUTRAL'. If NEUTRAL or BEAR, do not create a buy/half-position plan."
    )
    justification: str = dspy.OutputField(
        desc="Why this thesis was selected. Cite the specific data points FROM evidence_context that support it more strongly than the alternatives."
    )
    strongest_objection: str = dspy.OutputField(
        desc=(
            "The single strongest case AGAINST the selected thesis, stated in one to three sentences "
            "and drawn ONLY from evidence_context. Quote or name the specific evidence it rests on. "
            "If evidence_context contains nothing that argues against the selected thesis — including "
            "when the relevant sections are empty or unavailable — reply with exactly "
            "'INSUFFICIENT EVIDENCE' followed by what is missing. NEVER manufacture a plausible-sounding "
            "objection to fill this field, and never cite a figure, event, or date that is not in evidence_context."
        )
    )
    confidence_level: str = dspy.OutputField(
        desc="Confidence level: 'HIGH' (strong data alignment), 'MEDIUM' (mixed signals), or 'LOW' (conflicting data)"
    )
    confidence_reasoning: str = dspy.OutputField(
        desc="Why this confidence level was assigned. What would increase or decrease confidence?"
    )
    stop_loss: str = dspy.OutputField(
        desc="Suggested stop-loss only for a BULL thesis, anchored to support/ATR/technical structure from context. If unavailable or thesis is not BULL, use 'N/A'. Do not invent round-number stops."
    )
    profit_target: str = dspy.OutputField(
        desc="Suggested take-profit only for a BULL thesis using evidence from context. For SELL/NEUTRAL thesis, use 'N/A'."
    )
    action_summary: str = dspy.OutputField(
        desc="Final recommendation in one sentence. If BULL, include entry, structural stop, and target when available. If NEUTRAL/BEAR, state wait/avoid/sell logic with no entry blueprint."
    )

class ActiveThesisExtraction(dspy.Signature):
    """
    Extract active investment thesis details from a conversation.
    STRICT RULE: NEVER invent or hallucinate thesis details, targets, or dates not explicitly stated.
    """
    conversation_context: str = dspy.InputField(
        desc="The recent conversation history containing investment advice."
    )
    symbol: str = dspy.OutputField(
        desc="Stock ticker symbol (e.g. 'AAPL', 'VET.TO')."
    )
    action: str = dspy.OutputField(
        desc="Recommended action: 'BUY', 'SELL', 'HOLD', 'AVOID'."
    )
    quantity: str = dspy.OutputField(
        desc="Number of shares or contracts mentioned (e.g. '100', '50 shares'), or 'N/A' if none."
    )
    catalyst: str = dspy.OutputField(
        desc="Key upcoming event driving the thesis (e.g. 'Q1 Earnings')."
    )
    catalyst_date: str = dspy.OutputField(
        desc="Date of the catalyst if mentioned (YYYY-MM-DD), else 'N/A'."
    )
    stop_loss: str = dspy.OutputField(
        desc="Stop loss price or condition (e.g. '$12.00')."
    )
    target_price: str = dspy.OutputField(
        desc="Target exit price or profit goal (e.g. '$20.00' or '25% gain')."
    )
    conditions: str = dspy.OutputField(
        desc="Conditional logic (e.g. 'IF eps > 0.08 THEN hold ELSE sell')."
    )
    expiry_date: str = dspy.OutputField(
        desc="When this thesis expires (usually day after catalyst), or 'N/A'."
    )
    notes: str = dspy.OutputField(
        desc="Brief context/rationale (max 200 chars)."
    )


class FuturePrediction(dspy.Signature):
    """
    Project future market scenarios based on historical patterns and current regime.
    STRICT RULE: Provide projections based ONLY on historical matches. Do not invent false historical events or data.
    """
    current_context = dspy.InputField(
        desc="Current macro data (Inflation, Rates) and Market Trend"
    )
    historical_match = dspy.InputField(
        desc="The most similar historical period (e.g. '1995 Soft Landing') and its outcome"
    )

    forecast_3mo = dspy.OutputField(
        desc="Likely market direction over next 3 months (e.g. 'Bullish to 5200')"
    )
    forecast_1yr = dspy.OutputField(
        desc="Likely market direction over next 12 months"
    )
    portfolio_impact = dspy.OutputField(
        desc="Specific impact on the USER'S portfolio (Projected Value, Winners/Losers)."
    )
    confidence_score = dspy.OutputField(
        desc="Confidence score (0-100%) based on data alignment"
    )
    key_risks = dspy.OutputField(
        desc="Primary risks that could invalidate the forecast"
    )
    scenario_description = dspy.OutputField(
        desc="Narrative description of the predicted future (e.g. 'The Goldilocks Scenario')"
    )
