# pj-b-research-prompt-injection-attack

## Overview

This project aims to reproduce experiments from the paper, "Formalizing and Benchmarking Prompt Injection Attacks and Defenses," by Yupei Liu et al. The purpose is to systematically evaluate prompt injection attacks on Large Language Model (LLM)-integrated applications and explore different defensive strategies.

## Background

Prompt injection attacks involve injecting malicious instructions or data into the prompt of an LLM-integrated application to manipulate its output. The paper introduces a framework to formalize prompt injection attacks and conducts a comprehensive benchmark of five different attack methods and ten defensive strategies across multiple LLMs and tasks.

### About Experiment
1. select target_task, injected_task, prompt_num, and create prompt sets.
    - kind of task
        - DSD : duplicate sentense detection
        - GC : grammer correction
        - HD : hate detection 
        - NLI : natural language analysis
        - SA : sentiment analysis
        - SD : spam detection
        - Summ : summary
    - kind of attack
        - naive attack
        - excape characters
        - context ignoring
        - fake completion
        - combined attack

2. select LLM-model and process the prompt sets
    - kind of LLM-model
        - gemini
            - gemini-1.5-flash
            - gemini-1.5-pro

3. evaluate attacks
    - kind of indicator
       - ASV : Attack Success Value
       - MR : Matching Rate
       - PNA : Performance under No Attacks

