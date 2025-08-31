import os

from openai import OpenAI

"""
eg7.py — Formatting Fiesta demo: default vs. style-directed output on DeepSeek V3.

WHY this example:
- V-series models (especially V3) sometimes overuse italics/bold and produce choppy, single-sentence paragraphs by default.
- A simple system prompt that sets style expectations often yields smoother, connected prose.

What this script does:
1) Sends a topic prompt to V3 with no formatting guidance.
2) Sends the same prompt with a system message that requests flowing paragraphs and avoids typographical fireworks.
3) Prints both outputs so you can compare the difference.

Environment:
- Requires OPENROUTER_API_KEY in your environment.
- Uses OpenRouter to access the DeepSeek V3.1 model, consistent with other examples in this chapter.
"""

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "deepseek/deepseek-chat-v3.1:free"

# Topic prompt chosen to elicit multi-paragraph explanatory writing,
# where default V3 behavior may produce over-markdown'd or choppy text.
TOPIC_PROMPT = (
    "In 3–5 paragraphs, explain how to write effective internal AI usage guidelines "
    "for a product team. Include tradeoffs, pitfalls to avoid, and one short checklist."
)

# Explicit formatting spec to tame markdown-heavy or choppy output.
# Keep this crisp; models follow constraints better when phrased as clear rules.
SYSTEM_FORMATTING_SPEC = """Write in flowing, connected paragraphs. Avoid:
- Single-sentence paragraphs unless for emphasis
- Excessive use of *italics* or **bold** formatting
- Breaking thoughts into choppy segments
"""


def run_default(prompt: str) -> str:
    """
    Ask V3 without any style constraints.
    WHY: Serves as a baseline to compare how formatting can drift without guidance.
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def run_with_format_spec(prompt: str, system_spec: str) -> str:
    """
    Ask V3 with a system message that sets formatting expectations.
    WHY: System role is the right place to impose global style rules reliably.
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_spec},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print("=== DeepSeek V3 default output (no style guardrails) ===\n")
    default_output = run_default(TOPIC_PROMPT)
    print(default_output)

    print("\n\n=== DeepSeek V3 with explicit formatting spec ===\n")
    formatted_output = run_with_format_spec(TOPIC_PROMPT, SYSTEM_FORMATTING_SPEC)
    print(formatted_output)

"""
=== DeepSeek V3 default output (no style guardrails) ===

Of course. Here is an explanation of how to write effective internal AI usage guidelines for a product team, including tradeoffs, pitfalls, and a checklist.

### **Crafting Effective Guidelines**

Effective internal AI guidelines are not a list of prohibitions but a strategic framework that empowers product teams to innovate responsibly. Start by establishing the "why"—connecting the use of AI directly to the company's core values, product principles, and legal obligations. This foundation should address key pillars like user privacy, data security, fairness (mitigating bias), and transparency. The guidelines must then move from the abstract to the practical, providing clear, actionable directives. This includes specifying which AI models and tools are approved for use, defining strict protocols for handling user data (e.g., never sending PII to public APIs), and mandating a human-in-the-loop process for high-stakes decisions. Crucially, the document should be a living resource, offering examples of good and bad prompts, links to approved vendors, and a clear path for team members to get their specific use cases reviewed and approved.

### **Tradeoffs and Pitfalls to Avoid**

A significant tradeoff exists between **innovation speed and risk mitigation**. Overly restrictive guidelines can stifle experimentation and put the team at a competitive disadvantage. Conversely, overly lax policies create massive legal, reputational, and security risks. The goal is to find a balance by tiering guidelines based on risk; for example, using AI for internal code documentation requires far less scrutiny than using it for user-facing content generation. Common pitfalls include **writing in a vacuum** without input from legal, security, and engineering, leading to unenforceable rules. Another is creating a **static document** that doesn't evolve with the technology, quickly rendering it obsolete. Perhaps the most critical pitfall is **focusing only on the "build"** and ignoring the "buy"; guidelines must also cover the ethical procurement and integration of third-party AI-powered SaaS tools, which carry their own data risks.

### **Short Checklist for AI Usage**

Before using AI in a product workflow, ensure you can answer "yes" to the following:
*   [ ] **Data Check:** No sensitive, proprietary, or user PII is being submitted to the model.
*   [ ] **Quality Assurance:** All AI-generated output (code, text, design) is rigorously validated and edited by a human expert.
*   [ ] **Transparency:** The user is made aware they are interacting with AI (where appropriate).
*   [ ] **Bias Review:** The output has been checked for unfair bias or stereotyping.
*   [ ] **Legal Compliance:** The use case complies with relevant licenses, copyright, and regulatory requirements.


=== DeepSeek V3 with explicit formatting spec ===

Effective internal AI usage guidelines for a product team should begin by establishing clear principles that align with both the organization’s values and practical product goals. These principles might include prioritizing user benefit, ensuring transparency, maintaining data privacy, and committing to fairness and accountability. By grounding the guidelines in a shared ethical and operational framework, teams can make consistent decisions when integrating AI features. It’s also important to define scope: clarify which types of AI tools (e.g., generative AI, predictive models, or computer vision) the guidelines cover and specify relevant use cases, such as content generation, user personalization, or data analysis. This foundational clarity helps product managers, designers, and engineers understand the “why” behind the rules, fostering buy-in and thoughtful application.

When drafting these guidelines, it’s crucial to acknowledge key tradeoffs. For instance, striving for perfect model explainability might slow down development and time-to-market, so teams must balance transparency with practicality. Similarly, stringent data anonymization measures could improve privacy but may reduce the personalization that makes a product compelling. Another common tradeoff involves automation versus human oversight: fully automated AI features might scale efficiently but risk errors or misuse, while requiring human review can increase reliability at the cost of speed and resources. Recognizing these tensions allows teams to make informed, context-sensitive choices rather than applying rules rigidly.

Common pitfalls to avoid include being overly restrictive, which can stifle innovation, or too vague, which leads to inconsistent implementation. Guidelines should also avoid treating AI as a monolith; different applications require tailored rules. Finally, ensure the document is living—regularly updated as technology and regulations evolve. Below is a short checklist to support implementation:

- [ ] Define clear ownership and review processes for AI projects  
- [ ] Establish testing protocols for bias, accuracy, and safety  
- [ ] Include steps for user communication and consent  
- [ ] Specify data handling and security requirements  
- [ ] Plan for monitoring, feedback loops, and iterative improvement  

By combining principled guidance with pragmatic flexibility, product teams can harness AI’s potential responsibly and effectively.
"""
