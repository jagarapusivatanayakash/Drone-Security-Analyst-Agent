"""
Video summary prompts for generating comprehensive surveillance summaries
"""

VIDEO_SUMMARY_PROMPT_TEMPLATE = """Generate a comprehensive security summary for this surveillance video:

**Video Statistics:**
- Total Frames Processed: {total_frames}
- Total Alerts Generated: {total_alerts}

**Key Events (Sampled):**
{frames_summary}

**Alerts:**
{alerts_summary}

Provide a 3-5 sentence executive summary highlighting:
1. Overall security status
2. Key events and patterns
3. Notable alerts or concerns
4. Recommendations for follow-up
"""
