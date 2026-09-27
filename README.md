---
title: AutoML Studio
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AutoML Studio

Automated Machine Learning pipeline supporting dataset upload, automatic data cleaning, preprocessing, multi-model training (LightGBM, XGBoost, Random Forest, Logistic Regression, etc.), performance leaderboards, and instant inference.

## Run Locally with Docker

```bash
docker build -t aiml-auto .
docker run -p 7860:7860 aiml-auto
```
