# PhysicalAI Studio: Summer Plan for Paper Submission

Target submission deadlines:

-   ICRA: 16^th^ September

-   ICLR: 25^th^ September

-   CVPR: 14^th^ November

## Purpose

This document is intended to align the team on what we need to build,
improve, and validate over the summer to make **PhysicalAI
Studio** ready for a paper submission.

The emphasis is on identifying the core areas that support both:

-   a compelling research paper, and

-   a useful, software platform.

The structure below is designed so each section can be discussed
independently with the team on their respective specialties.

## Discussion points for the team

For each section above, we should align on:

-   what is essential for the paper,

-   what is feasible over the summer,

-   what has the highest leverage,

-   and who should own it.

## Data and datasets

### Motivation

If PhysicalAI Studio is meant to support policy training, evaluation,
and deployment across robotics workflows, then we need a clear and
consistent story for how data is collected, represented, stored, loaded,
and reused.

A strong data layer would support imitation learning, RL rollouts,
DAGGER-style aggregation, and world model training, while making it
easier to compare methods consistently.

### To implement

1.  Observation refactor we need to make use of Observation across the
    packages; do we use batch first always? Do we even use it at all? Is
    numpy a more effective tool to pass data around?

2.  Decide how to represent:

    a.  RL data format

    b.  DAGGER-collection data format

    c.  World model data format

    d.  Create example datasets or canonical reference datasets for
        internal use on huggingface

    e.  Add tooling needed for dataset versioning or data quality
        (datumaro?)

3.  Do we need any other dataset format other than LeRobotDataModule?

## Policies

### Motivation

Policies are central to the research and product value of the platform.
Enabling 1^st^ policies should be a priority to enable deployment.

This section should help answer whether the platform is only a wrapper
around existing methods, or whether it provides a coherent framework for
developing and evaluating policies across embodied AI tasks.

### To implement

-   Standardise interface for config, input / output features of
    Policies -- to help UI / export.

The list of 1^st^ party RL, World and VLA models to implement 1^st^
party (can be changeable):

  -----------------------------------------------------------------------
  MolmoAct 2
  -----------------------------------------------------------------------
  RLDX-1

  DreamZero

  VLA-JEPA

  Spirit v1.5

  LingBot-VLA

  Qwen-VLA
  -----------------------------------------------------------------------

For a more detailed list, reader can refer to this excel
[sheet](https://intel-my.sharepoint.com/:x:/p/samet_akcay/IQDD_l82x1_fQ6HCf-bRDq5cAbJDy_bhaqAv7wJQdnR7vUE?e=e3S4VU)

## Runtime

### Motivation

For PhysicalAI Studio to feel like a serious robotics/embodied AI
platform, it needs a clean path from model development to inference and
execution in real settings. This is especially valuable if we want to
distinguish PhysicalAI Studio from frameworks that stop at training.

### To implement

-   More backends

    -   ExecuTorch, TensorRT?

-   Solidify manifest for pre / post processing

-   Add more robot definitions

-   Nodes for ROS2, ZeroMQ etc?

## Quantization + tips / tricks

### Motivation

Quantization and deployment optimization are important if we want the
software to be practical in edge inference settings. This section is
also useful because it can capture practical engineering knowledge that
often makes the difference between a system that works in demos and one
that works reliably in practice.

### To implement

-   Add or improve quantization support for relevant policy/runtime
    paths

-   Document recommended quantization workflows

<!-- -->

-   Identify model classes that benefit most from quantization

-   Benchmark latency or efficiency improvements where feasible

## Hardware (Robots + Cameras)

### Motivation

Hardware support is important for credibility, especially if we want
PhysicalAI Studio to be viewed as a real embodied AI platform rather
than only a simulation framework. Even a limited but strong hardware
story can significantly strengthen the paper.

### To implement

-   Prioritize supported robot platforms

    -   Seeed robot arm

    -   Unitree R1-D

-   Identify missing hardware-specific modules or connectors

## Benchmarks

### Motivation

Benchmarks are essential for the paper because they provide the evidence
behind our claims. Without a clear evaluation framework, the platform
risks appearing as a collection of features rather than a validated
system.

Overall feeling is to implement 1-3 1^st^ party benchmarks and then
leave the heavy benchmarks for 3^rd^ party in paper submissions. For
real hardware we can set benchmarks based on our hardware -- which
hopefully should be diverse.

### To implement

For the paper we must decide on a set of simulated and real-world
benchmarks:

-   Sim

    -   LIBERO, Metaworld, Simpler-Env, Robosuite

-   Heavier weight

    -   Isaac-sim

-   Real hardware: on each of our robots we must choose a task that
    reflects the purpose of its construction

    -   SO-101: simpler cleaning or moving items

    -   Trossen: with two arms perhaps more tactile like folding clothes

    -   Seeed studio: like SO-101 but with more torque?

    -   Unitree: even more tactile than trossen perhaps screwing a bolt
        onto a nut.

-   For the above we freeze extra dataset collection past a certain date
    and run full suite of models before paper.

## UI / Data collection improvements

### Motivation

UI and data collection improvements are important because they can
materially improve the usability of the platform and reduce friction in
generating the data needed for training and evaluation. This is
especially relevant if one of the strengths of PhysicalAI Studio is that
it supports end-to-end experimentation.

### To implement

-   Improve tooling for reviewing and merging collected data

-   Document the intended collection workflow

-   Add import / export functionalities to include any huggingface model
    / dataset

-   Define UI look for paper -- it will be our branding.

## Resources

-   **RoboArena** --- robochallenge.ai/leaderboard (real-world,
    crowd-sourced pairwise Elo; primary frontier signal)

-   **RoboChallenge / Table30** --- robochallenge.cn/home (real-world,
    standardized 30 tasks, absolute success rate)

-   **MolmoSpaces** --- molmospaces.allen.ai/leaderboard (Ai2, real+sim,
    open/reproducible)

-   **LIBERO-Plus / LIBERO-PRO** --- github.com/sylvestf/LIBERO-plus
    (sim robustness, not saturated standard LIBERO)

-   **CVPR Embodied-AI workshop challenges** ---
    navigation/rearrangement (AI2-THOR, Habitat)

-   **NeurIPS BEHAVIOR Challenge** --- behavior.stanford.edu/challenge
    (long-horizon household)

-   <https://github.com/keon/awesome-physical-ai>

-   <https://github.com/natnew/awesome-physical-ai>