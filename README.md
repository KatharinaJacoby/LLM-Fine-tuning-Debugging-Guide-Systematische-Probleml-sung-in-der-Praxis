# Fixing T5 Fine-Tuning Bugs for Medical Diagnosis

This notebook documents the debugging process of fine-tuning T5-small for medical diagnosis tasks. What started as a simple text-to-text classification project turned into a deep dive into tokenizer handling, generation bugs, and training instabilities.

## Problem
After training, the model only output `"True"` or repeated inputs. Despite a great loss curve, the outputs were useless.

## Method
I approached this by:
- Isolating tokenizer and data collator issues
- Testing task prefix impact (critical for T5!)
- Comparing fresh vs continued training
- Tuning generation parameters (e.g. beam search, repetition penalty)

## Results
Final loss: **0.009**, 100% accuracy on handcrafted test set, and clear medical diagnosis outputs like `"Pneumonia"` or `"Myocardial infarction"`.

www.kaggle.com/code/kjacoby/debugging-guide-t5-fine-tuning-true-bug
