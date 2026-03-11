# Project Payphone – Storytelling LLM Specification

## Objective

Build a specialized language model that converts historical Old Bailey court records into short pieces of legal historical fiction. The generated output should be a coherent narrative story rather than a summary of the case.

## Core Output Requirements

- **Story length:** 400–600 words
- **Format:** continuous prose (no headings, bullet points, or stage labels in the final output)
- **Narrative arc:** complete beginning, middle, and end
- **Structural framework:** Joseph Campbell / Jungian Hero's Journey
- **Genre:** historical crime fiction inspired by real legal records
- **Tone:** flexible based on generation mode (dark, victim-centered, sympathetic defendant, etc.)

## Narrative Constraints

The Hero's Journey must act as the internal structure guiding the narrative. The model may internally plan using Hero's Journey stages but must output only natural prose.

Typical narrative flow should resemble:

1. **Ordinary world** – setting and context (London, shop, street life)
2. **Call to adventure** – the crime or disturbance occurs
3. **Threshold** – discovery or pursuit begins
4. **Trials** – witnesses, confusion, or chase
5. **Ordeal** – confrontation, arrest, or trial
6. **Resolution** – judgement and punishment
7. **Reflection** – moral or social observation

## Historical Anchors

The following elements must remain consistent with the source record:

- crime / offence
- victim
- verdict
- punishment
- general event description

## Creative Expansion

The model is allowed to invent details to make the narrative compelling. Acceptable invented elements include:

- dialogue
- character motivations
- bystanders or witnesses
- environmental description
- internal thoughts
- minor events between recorded facts

This controlled hallucination is intentional and required for storytelling quality.

## Input Data Source

Stories are generated from Old Bailey court records stored in local SQLite databases.

Example input fields:

- year
- offence
- defendant
- victim
- verdict
- punishment
- full case text

## System Architecture

The system currently includes:

### Flask Application

- Case browser interface
- Allows browsing offences and speeches from Old Bailey records

### FastAPI Service

- Provides case listing and case detail endpoints
- Generates legal fiction stories using a local LLM

### LLM Runtime

- Ollama local inference
- Current default model: Mistral 7B

### Databases

- **oldbailey.sqlite** – used by Flask browser
- **old_bailey.db** – used by FastAPI generation pipeline

## Future Model Training Goal

Fine-tune an open-source LLM so it specializes in:

**Old Bailey case record → structured historical narrative**

The model should learn to automatically:

- map case facts into a Hero's Journey narrative arc
- generate compelling historical prose
- produce consistent 400–600 word stories

## Training Method

- LoRA / QLoRA fine-tuning
- Base models considered: Mistral-7B or Qwen2.5-7B
- Training dataset derived from Old Bailey cases paired with generated stories

## Generation Modes

The API supports different narrative perspectives:

- **dark** – fatalism, brutality, bleak atmosphere
- **sympathetic_defendant** – structural injustice, humanizing defendant
- **victim_centered** – harm, fear, aftermath, victim dignity
- **courtroom_focused** – rhetoric, exchanges, procedural tension
- **pamphlet_style** – 17th-century moralizing pamphlet voice

These modes influence tone and narrative perspective but do not change the Hero's Journey structure.

## Project Goal

Create a specialized storytelling LLM capable of turning historical legal records into engaging narrative crime stories while preserving the core historical facts.
