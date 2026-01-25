# Role-Stratified-LLM-Analysis-in-Social-Media-Cancer-Narrative-Weibo

This repository provides a **sanitized, reproducible reference implementation** of the data processing and annotation pipeline used in our study on large-scale cancer-related narratives on Weibo (2020–2025). It covers:

- two-stage supervised screening (speaker attribution + authenticity verification)
- BERTopic-based macro thematic scaffolding
- LLM-assisted multidimensional labeling (DeepSeek API)
- structured coding (Event Primary/Secondary, Domain, Meaning Making)
- clinically oriented classification for high-intensity posts (clinical_triage + ESG-6 handling)
- downstream visualization and analysis utilities

> **Privacy note (important):** This repository does **not** contain raw Weibo text, user identifiers, or any traceable personal information. The released code is structured to run on **your own locally stored dataset** after you complete your institutional/ethical compliance steps.

---

