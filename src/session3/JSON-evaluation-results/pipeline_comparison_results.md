# Pipeline Comparison Results

Existing method: 5 subset drafts.
Cluster-direct method: 2 drafts, one per cluster.
Completed comparisons: 6 of 12 ground-truth questions.

## Summary

| Method | Final accuracy | Average draft and verifier seconds |
|---|---|---|
| 5 subset drafts | 100.0% | 59.7 |
| 2 cluster drafts | 100.0% | 43.4 |

## Results by question

### E02: Which AWS service handles user authentication in the FitStyle AI platform?

Reference answer: Amazon Cognito

#### 5 subset drafts

- Final answer: Amazon Cognito is the AWS service that handles user authentication in the FitStyle AI platform.
- Correct: True
- **Draft 1** — verifier score: 10.0
  The AWS service that handles user authentication in the FitStyle AI platform is Amazon Cognito.
- **Draft 2** — verifier score: 10.0
  The Amazon Cognito service handles user authentication in the FitStyle AI platform.
- **Draft 3** — verifier score: 10.0
  Amazon Cognito is the AWS service that handles user authentication in the FitStyle AI platform.
- **Draft 4** — verifier score: 10.0
  The Amazon Cognito service handles user authentication in the FitStyle AI platform.
- **Draft 5** — verifier score: 10.0
  The FitStyle AI platform uses AWS Cognito to handle user authentication.

#### 2 cluster drafts

- Final answer: Amazon Cognito handles user authentication in the FitStyle AI platform.
- Correct: True
- **Draft 1** — verifier score: 10.0
  Amazon Cognito handles user authentication in the FitStyle AI platform.
- **Draft 2** — verifier score: 0.0


### E03: How many products does the FitStyle AI catalogue contain, and which clothing sizes does it stock?

Reference answer: 11,513 products, in sizes S to XL only

#### 5 subset drafts

- Final answer: The FitStyle AI catalogue contains 11,513 products and stocks clothing sizes ranging from S to XL.
- Correct: True
- **Draft 1** — verifier score: 10.0
  The FitStyle AI catalogue contains 11,513 products and stocks clothing sizes ranging from S to XL.
- **Draft 2** — verifier score: 10.0
  The FitStyle AI catalogue contains 11,513 products and stocks clothing sizes ranging from S to XL.
- **Draft 3** — verifier score: 10.0
  The FitStyle AI catalogue contains 11,513 products and it stocks clothing sizes ranging from S to XL.
- **Draft 4** — verifier score: 5.0
  The FitStyle AI catalogue contains products in sizes S to XL only.
- **Draft 5** — verifier score: 0.0
  The documents do not specify the total number of products in the FitStyle AI catalogue, but state that it stocks clothing sizes S to XL.

#### 2 cluster drafts

- Final answer: The FitStyle AI catalogue contains 11,513 products and stocks clothing sizes ranging from S to XL.
- Correct: True
- **Draft 1** — verifier score: 0.0
  The provided documents do not specify the number of products in the FitStyle AI catalogue or the specific clothing sizes it stocks.
- **Draft 2** — verifier score: 10.0
  The FitStyle AI catalogue contains 11,513 products and stocks clothing sizes ranging from S to XL.

### E04: On what kind of GPU runtime does the virtual try-on model run in this project?

Reference answer: a free Google Colab T4 GPU

#### 5 subset drafts

- Final answer: The virtual try-on model, identified as Model 3 using LaDI-VTON, runs on a free GPU runtime provided by a Colab T4 GPU.
- Correct: True
- **Draft 1** — verifier score: 3.0
  The virtual try-on model in this project runs on a free GPU as a queued service.
- **Draft 2** — verifier score: 10.0
  The virtual try-on model runs on a Google Colab environment utilizing a T4 GPU.
- **Draft 3** — verifier score: 10.0
  The virtual try-on model, identified as LaDI-VTON within Model 3, runs on a Google Colab environment equipped with a T4 GPU.
- **Draft 4** — verifier score: 10.0
  The virtual try-on model, identified as Model 3 using LaDI-VTON, runs on a free GPU runtime provided by a Colab T4 GPU.
- **Draft 5** — verifier score: 10.0
  The virtual try-on model in this project runs on a free Colab T4 GPU.

#### 2 cluster drafts

- Final answer: The virtual try-on model (Model 3) runs as a separate service in a notebook on a free Colab T4 GPU.
- Correct: True
- **Draft 1** — verifier score: 10.0
  The LaDI-VTON virtual try-on model (Model 3) runs on a free Colab T4 GPU.
- **Draft 2** — verifier score: 10.0
  The virtual try-on model (Model 3) runs as a separate service in a notebook on a free Colab T4 GPU.

### M01: By how much did rebalancing the compatibility network improve held-out accuracy? Give the accuracy of CompatibilityNet v1 and v2.

Reference answer: accuracy rose from 0.646 (v1) to 0.789 (v2)

#### 5 subset drafts

- Final answer: Rebalancing the compatibility network improved held-out accuracy from 0.646 (CompatibilityNet v1) to 0.789 (CompatibilityNet v2).
- Correct: True
- **Draft 1** — verifier score: 10.0
  Rebalancing the compatibility network improved held-out accuracy from 0.646 (CompatibilityNet v1) to 0.789 (CompatibilityNet v2).
- **Draft 2** — verifier score: 8.0
  Rebalancing the compatibility network improved the held-out accuracy from 0.646 to 0.789.
- **Draft 3** — verifier score: 0.0
  The documents do not contain the specific accuracy values for CompatibilityNet v1 and v2, nor do they state by how much rebalancing improved held-out accuracy; however, they note that the metadata-only model was almost as accurate as the linear model with visual features, suggesting proxy labels depend mainly on category and color.
- **Draft 4** — verifier score: 10.0
  Rebalancing the compatibility network improved held-out accuracy from 0.646 for CompatibilityNet v1 to 0.789 for CompatibilityNet v2.
- **Draft 5** — verifier score: 0.0
  The provided documents state that the rebalanced network improved on its first version but do not contain the specific accuracy figures for CompatibilityNet v1 or v2.

#### 2 cluster drafts

- Final answer: Rebalancing the compatibility network improved held-out accuracy from 0.646 (CompatibilityNet v1) to 0.789 (CompatibilityNet v2).
- Correct: True
- **Draft 1** — verifier score: 10.0
  Rebalancing the compatibility network improved held-out accuracy from 0.646 (CompatibilityNet v1) to 0.789 (CompatibilityNet v2).
- **Draft 2** — verifier score: 0.0
  Rebalancing the compatibility network did not improve held-out accuracy because CompatibilityNet v2 had an accuracy of 0.789 while the linear model achieved 0.932.

### M02: What weight did the validation split select for the CompatibilityNet in the ensemble, and what does that mean for the final model?

Reference answer: the selected weight was 0, so the ensemble reduces to the linear model alone

#### 5 subset drafts

- Final answer: The validation split selected a weight of 0 for the CompatibilityNet in the ensemble, meaning that the final model effectively reduces to the linear model because the neural network component contributed no value.
- Correct: True
- **Draft 1** — verifier score: 10.0
  The validation split selected a weight of 0 for the CompatibilityNet in the ensemble, meaning that the final model effectively reduces to the linear model because the neural network component contributed no value.
- **Draft 2** — verifier score: 0.0
  The documents do not state that the validation split selected a weight for the CompatibilityNet in the ensemble, as the text indicates the default weighting is a stated design preference and not an empirically validated optimum.
- **Draft 3** — verifier score: 0.0
  The documents do not specify the exact weight value selected for the CompatibilityNet in the ensemble, only that the weight was chosen from a grid between 0 and 1 to maximize accuracy on the validation split.
- **Draft 4** — verifier score: 10.0
  The validation split selected a weight of 0 for the CompatibilityNet in the ensemble, meaning the final model effectively becomes the linear model. (Document [1])
- **Draft 5** — verifier score: 8.0
  The validation split selected full weight for the linear model, meaning the CompatibilityNet was effectively excluded from the final ensemble in favor of the simpler linear approach.

#### 2 cluster drafts

- Final answer: The validation split selected a weight of 0 for the CompatibilityNet in the ensemble, meaning the final model reduces to the linear model.
- Correct: True
- **Draft 1** — verifier score: 10.0
  The validation split selected a weight of 0 for the CompatibilityNet in the ensemble, meaning the final model reduces to the linear model.
- **Draft 2** — verifier score: 9.0
  The validation split selected full weight for the linear model, meaning the CompatibilityNet received zero weight and the final ensemble reduced to the linear model alone.

### M04: How many profiles are pre-computed offline for the recommendation engine, and why is this done instead of running the model live?

Reference answer: 400 profiles (4 sizes x 5 body shapes x 5 occasions x 4 seasons); the neural pipeline needs a GPU and the application should stay within free-tier limits

#### 5 subset drafts

- Final answer: 400 profiles are pre-computed offline for the recommendation engine because the neural pipeline requires a GPU and the application should stay within free-tier limits.
- Correct: True
- **Draft 1** — verifier score: 0.0
  The documents do not specify how many profiles are pre-computed offline for the recommendation engine.
- **Draft 2** — verifier score: 0.0
  Four profiles (stored in four existing DynamoDB tables) are pre-computed offline for the recommendation engine because those tables already consume the available free-tier read capacity.
- **Draft 3** — verifier score: 0.0
  The documents do not specify the number of pre-computed profiles, but state that the recommendation engine is pre-computed to ensure no neural inference occurs at request time.
- **Draft 4** — verifier score: 10.0
  400 profiles are pre-computed offline for the recommendation engine because the neural pipeline requires a GPU and the application should stay within free-tier limits.
- **Draft 5** — verifier score: 8.0
  The recommendation engine pre-computes 400 profiles offline to avoid running the neural pipeline on every request. This information comes from document [3]. The rationale is that this approach serves recommendations without executing the neural inference at request time.

#### 2 cluster drafts

- Final answer: 400 profiles are pre-computed offline because the neural pipeline requires a GPU and the application should stay within free-tier limits.
- Correct: True
- **Draft 1** — verifier score: 0.0
  The documents do not state how many profiles are pre-computed offline, but they explain that this approach is used because the existing four tables already consume the available free-tier read capacity, making it the main reason for pre-computing recommendations rather than storing them in a new table or running inference live.
- **Draft 2** — verifier score: 10.0
  400 profiles are pre-computed offline because the neural pipeline requires a GPU and the application should stay within free-tier limits.
