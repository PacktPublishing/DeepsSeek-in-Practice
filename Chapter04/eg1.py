import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

PROMPT = "Is it ethically OK for a hospital to use a black-box AI triage system in the ER if it’s more accurate than doctors? Please discuss pros and cons and conclude with what you’d recommend."

completion = client.chat.completions.create(
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
            ],
        }
    ],
)
print(completion.choices[0].message.content)

"""MISTRAL OUTPUT:

The ethical implications of using a black-box AI triage system in the Emergency Room (ER), even if it proves to be more accurate than doctors, is a complex issue that involves balancing clinical efficacy, patient trust, transparency, and accountability. Here’s a breakdown of the pros and cons, followed by a recommendation:

### **Pros:**
1. **Improved Accuracy and Efficiency:**
   - AI systems can process vast amounts of data quickly and may detect patterns that humans miss, leading to faster and more accurate triage decisions.
   - Reduced human error, especially in high-pressure ER environments where fatigue or cognitive biases might affect decision-making.

2. **Consistency and Scalability:**
   - AI can provide uniform assessments across different locations and healthcare providers, reducing variability in triage quality.
   - Useful in understaffed or overcrowded ERs where human resources are stretched thin.

3. **Early Warning Systems:**
   - AI could flag critical cases earlier, potentially improving patient outcomes in life-threatening situations.

4. **Data-Driven Decision Making:**
   - AI can incorporate the latest medical research and guidelines into its decision-making process, ensuring that triage aligns with best practices.

### **Cons:**
1. **Lack of Transparency (Black-Box Problem):**
   - If the AI’s decision-making process is not explainable, clinicians and patients may struggle to trust its recommendations.
   - Without transparency, it’s difficult to identify biases in the AI system or understand why certain decisions were made.

2. **Accountability and Liability:**
   - If an AI-driven triage error leads to harm, determining liability (whether the hospital, AI developers, or clinicians are responsible) can be legally and ethically challenging.

3. **Over-reliance on Technology:**
   - Clinicians may become overly dependent on AI, potentially leading to deskilling or misses in critical judgment.

4. **Bias and Fairness:**
   - If the AI was trained on biased data (e.g., underrepresenting certain demographic groups), it could perpetuate or exacerbate healthcare disparities.

5. **Patient Trust and Autonomy:**
   - Patients may feel uncomfortable with an AI making critical decisions about their care without human override or explanation.

6. **Regulatory and Legal Challenges:**
   - Many countries lack clear guidelines on AI deployment in healthcare, raising concerns about compliance with medical standards.

### **Recommendation:**
While AI-driven triage has significant potential, its adoption in the ER should be approached cautiously. Here’s what I recommend:
1. **Transparency and Explainability:**
   - Hospitals should prefer AI systems that provide clear, interpretable reasoning for their triage decisions rather than using purely "black-box" models.

2. **Human-in-the-Loop Model:**
   - AI should assist rather than replace clinicians. Final decisions should involve human oversight, especially in ambiguous cases.

3. **Transparent Testing and Validation:**
   - Rigorous clinical trials should prove the AI’s effectiveness and safety before widespread adoption, ensuring it performs well across diverse patient populations.

4. **Ethical and Legal Safeguards:**
   - Hospitals must establish clear accountability frameworks and ensure compliance with patient privacy laws (e.g., HIPAA in the U.S.).

5. **Patient Consent and Education:**
   - Patients should be informed about AI involvement in their care and have the option to request human review if desired.

6. **Continuous Monitoring:**
   - AI systems should be frequently audited for biases, errors, and performance issues.

### **Conclusion:**
AI triage in the ER holds promise but must be implemented ethically and responsibly. If transparency, accountability, and human oversight are prioritized, such systems could be a valuable tool—**provided they are used as a supplement, not a replacement, for human judgment.**
"""
