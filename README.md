# SOTIF-Compliant Framework for Scenario Generator Evaluation in CARLA

## Overview

This project implements a **SOTIF-aligned evaluation framework** for the empirical comparison of automatic scenario generation tools used in the validation of **Advanced Driver Assistance Systems (ADAS)** and **Automated Driving Systems (ADS)** in simulation.

The main goal is not to evaluate a single autonomous driving system in isolation, but to provide a **structured and reproducible methodology** for comparing different scenario generators according to criteria that are meaningful from a **safety-of-the-intended-functionality** perspective. In particular, the framework is designed to support the analysis of:

- the capability of a generator to produce **hazardous scenarios**;
- the coverage of the **Operational Design Domain (ODD)** and related **triggering conditions**;
- the **diversity** of the hazardous situations discovered;
- the overall **efficiency** of the generation-and-evaluation process.

The project is grounded in the principles of **ISO 21448:2022 (SOTIF)**, which focuses on hazardous behaviors caused not by faults in the system, but by functional insufficiencies or foreseeable limitations of the intended functionality.

## Motivation

Testing ADS/ADAS exclusively in the real world is not sufficient for a rigorous safety assessment. Rare but safety-critical events are difficult, expensive, and often unsafe to reproduce on public roads. For this reason, simulation environments such as **CARLA** play a central role in modern validation workflows.

However, manually crafting scenarios for simulation is time-consuming and difficult to scale. Automatic scenario generators address this limitation by producing test scenarios in a systematic way. The problem is that, without a standard-oriented evaluation methodology, it is hard to understand **which generator is actually more useful for safety validation**.

This project addresses that gap by proposing a framework that evaluates scenario generators through a SOTIF-oriented lens, combining **risk-based assessment**, **ODD analysis**, and **scenario diversity analysis**.

## Project Objective

The objective of the framework is twofold:

1. **Empirical comparison of scenario generation tools** in a common evaluation setting;
2. **Demonstration of a reusable SOTIF-compliant methodology** for assessing the quality of generated scenarios in simulation.

The framework is designed to be as **generator-agnostic** as possible: the same evaluation logic can be applied to scenarios produced by different tools, provided that they are executed in a uniform simulation pipeline and logged in a compatible format.

## What the framework evaluates

The framework supports the analysis of multiple complementary dimensions:

### 1. Hazard effectiveness
It measures how effective a scenario generator is at exposing potentially unsafe behavior, such as collisions or traffic-rule violations, by aggregating simulation outcomes across repeated executions.

### 2. ODD and Triggering Condition coverage
Each generated scenario is analyzed with respect to ODD-related dimensions such as environmental, infrastructural, traffic, and operational factors. In addition, the framework derives **triggering conditions** that may activate hazardous system behavior under challenging circumstances.

### 3. Hazardous scenario diversity
The framework also investigates whether a generator discovers genuinely different hazardous situations or simply repeats variations of the same failure pattern. To support this analysis, feature vectors can be extracted from enriched execution logs for downstream clustering and diversity assessment.

### 4. Final SOTIF-oriented reporting
The different analysis steps are aggregated into final outputs that support the interpretation of the generator’s safety relevance in a structured way.

## Current pipeline structure

At the current stage, the project includes a **multi-dataset SOTIF pipeline**. For each dataset folder contained in `datasets/`, the pipeline performs the following macro-steps:

1. **Sanity check of base logs**
2. **Descriptive ODD analysis**
3. **Hazard and severity computation**
4. **Final SOTIF report generation**

The pipeline is designed to process multiple datasets in a uniform way, making it suitable for comparing outputs produced by different scenario generation tools under the same evaluation workflow.

## Conceptual workflow

In conceptual terms, the project follows this logic:

- scenario generators produce candidate driving scenarios;
- scenarios are executed in simulation;
- execution logs are collected in a common format;
- logs are enriched with ODD-related, hazard-related, and behavioral information;
- aggregated metrics are computed to support comparative analysis across generators.

This separation between **scenario generation**, **scenario execution**, and **post-execution SOTIF analysis** makes the framework modular and easier to extend.

## Why this repository matters

This repository is meant to serve as a practical implementation of a broader research effort on the standardized evaluation of scenario generators for autonomous driving validation. Rather than focusing only on raw failure discovery, the project aims to provide a more meaningful assessment based on:

- safety relevance,
- operational-context coverage,
- diversity of discovered hazards,
- reproducibility of the evaluation process.

In this sense, the repository is both a **research artifact** and a **reusable experimental pipeline** for future studies on scenario-based validation in CARLA.

## Requirements

The project has been developed and tested using the following environment:

- **Python version:** `3.7.16`
- **Operating system:** Linux (tested on Ubuntu)

Before running the pipelines, make sure that the correct Python version is available and that all required dependencies are installed.

### Create a virtual environment (recommended)

It is strongly recommended to run the project inside a virtual environment.

```bash
python3.7 -m venv venv
source venv/bin/activate
```

## Install project dependencies

Once the virtual environment is activated, install the required libraries using:

```bash
pip install -r requirements.txt
```

This will install all the dependencies needed to run the SOTIF evaluation pipeline and the analysis pipeline.
