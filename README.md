# pj-b-research-prompt-injection-attack

## Overview

This project aims to reproduce experiments from the paper, "Formalizing and Benchmarking Prompt Injection Attacks and Defenses," by Yupei Liu et al. The purpose is to systematically evaluate prompt injection attacks on Large Language Model (LLM)-integrated applications and explore different defensive strategies.

## Background

Prompt injection attacks involve injecting malicious instructions or data into the prompt of an LLM-integrated application to manipulate its output. The paper introduces a framework to formalize prompt injection attacks and conducts a comprehensive benchmark of five different attack methods and ten defensive strategies across multiple LLMs and tasks.

### Setup

### How To Use
1. select target_task, injected_task, prompt_num を選択してプロンプトを生成する
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

2. 

