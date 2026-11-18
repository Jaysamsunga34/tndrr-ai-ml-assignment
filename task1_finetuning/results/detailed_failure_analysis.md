# Task 1: Genuine Failure Analysis (Fine-Tuned Qwen2.5-0.5B-Instruct LoRA)

On the 68-sample leak-free held-out test set (which includes 14 adversarial edge cases), LoRA fine-tuning improved Exact JSON Match from **0.00% to 32.35%**, Category Accuracy from **20.59% to 75.00%**, Priority Accuracy from **51.47% to 77.94%**, and Intent Accuracy from **0.00% to 51.47%**, while preserving **100.0% JSON validity**.

Below are **four genuine, distinct failure modes** observed directly in `finetuned_predictions.csv` and `failure_analysis.csv`.

---

## Failure Case 1: Deceptive Urgency Negation (Prefix Attention Shortcut)
- **Input:** `"URGENT PLEASE READ: Actually I'm not in a hurry at all, whenever you have time next week could you share enterprise pricing for 50 seats?"`
- **Expected Output:** `{"category": "general", "priority": "low", "intent": "pricing_inquiry"}`
- **Actual Generated Output:** `{"category": "general", "priority": "high", "intent": "pricing_inquiry"}`
- **What Went Wrong:** Category (`general`) and intent (`pricing_inquiry`) were exact matches, but priority was misclassified as `high` instead of `low`.
- **Technical Explanation:** The opening all-caps prefix `"URGENT PLEASE READ:"` acted as a high-salience lexical shortcut in the low-rank (`r=8`) attention adapters, overriding the subsequent pragmatic negation (`"Actually I'm not in a hurry at all"`). In the training distribution, uppercase urgency headers correlated exclusively with `high`/`critical` priorities.
- **Potential Improvement:** Add hard-negative training examples featuring negated urgency keywords, and introduce a short reasoning field (or chain-of-thought distillation) before emitting the priority enum.

---

## Failure Case 2: Implicit Operational Severity & Active-vs-Retrospective Intent Confusion
- **Input:** `"An unknown user from a foreign IP is actively exporting our entire customer table from the admin panel as I type this!"`
- **Expected Output:** `{"category": "security", "priority": "critical", "intent": "account_takeover_suspected"}`
- **Actual Generated Output:** `{"category": "security", "priority": "medium", "intent": "data_breach_inquiry"}`
- **What Went Wrong:** Routed to the right category (`security`), but critically under-prioritized (`medium` instead of `critical`) and confused active ongoing account compromise with a retrospective breach inquiry.
- **Technical Explanation:** Unlike training `critical` examples that carried explicit framing cues (`"EMERGENCY:"`, `"Critical incident"`), this test case expressed severity purely through operational consequence (`"actively exporting our entire customer table"`). Simultaneously, `"customer table"` and `"exporting"` pulled the intent projection toward `data_breach_inquiry`.
- **Potential Improvement:** Remove explicit severity prefixes from a portion of training samples so the model learns consequence-based priority estimation, and apply asymmetric/cost-sensitive loss weighting for `critical` security classes.

---

## Failure Case 3: Out-of-Schema Intent Hallucination (`2fa_bypass`)
- **Input:** `"My phone fell in a lake so my Google Authenticator is gone and I have an investor demo in 20 minutes—help me bypass 2FA!"`
- **Expected Output:** `{"category": "account", "priority": "high", "intent": "2fa_trouble"}`
- **Actual Generated Output:** `{"category": "security", "priority": "high", "intent": "2fa_bypass"}`
- **What Went Wrong:** Correctly detected `high` priority, but routed to `security` instead of `account` and synthesized an unseen intent label (`"2fa_bypass"`) outside the 28-class taxonomy.
- **Technical Explanation:** The verb `"bypass"` activated pretrained security associations (e.g., authentication bypass vulnerabilities) and caused unconstrained autoregressive decoding to copy `"bypass"` directly into the snake_case intent string.
- **Potential Improvement:** Apply **constrained decoding** (JSON Schema / GBNF grammar enforcement over the closed set of 28 allowed intents) at inference time and enumerate canonical intents or include few-shot boundary examples in the prompt.

---

## Failure Case 4: Left-to-Right Autoregressive Key-Order Inconsistency
- **Input:** `"Following up with a routine support question: Stripe checkout rejects our corporate debit card every time we retry."`
- **Expected Output:** `{"category": "billing", "priority": "medium", "intent": "payment_failed"}`
- **Actual Generated Output:** `{"category": "technical", "priority": "medium", "intent": "payment_failed"}`
- **What Went Wrong:** Predicted both `priority` (`medium`) and `intent` (`payment_failed`) with 100% accuracy, yet predicted `category: "technical"` instead of `"billing"`—violating the deterministic taxonomy mapping (`payment_failed` $\rightarrow$ `billing`).
- **Technical Explanation:** Because JSON keys are emitted autoregressively left-to-right (`category` first, `intent` last), the model prematurely committed to `"technical"` based on integration terminology (`"Stripe checkout rejects"`) before resolving the finer-grained `"payment_failed"` intent later in the sequence.
- **Potential Improvement:** Reorder the output schema so `intent` is generated **first** (allowing `category` to condition directly on the predicted intent token), or derive `category` deterministically from `intent` in application post-processing.
