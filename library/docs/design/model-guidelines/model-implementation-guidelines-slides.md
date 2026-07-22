---
marp: true
theme: default
paginate: true
transition: fade
style: |
  section { font-size: 28px; }
  table { font-size: 21px; }
  h1, h2 { color: #0068b5; }
  section.lead h1 { font-size: 56px; }
---

<!-- _class: lead -->

# Physical AI Model Enablement

## Broad coverage, supported workflows, optimized Intel deployment

---

## Decision

Studio needs two outcomes:

- Make useful models available quickly.
- Deliver production-quality Intel deployment for selected models.

Do not report both as one model count.

---

## Four Independent Attributes

| Attribute              | Answers                      |
| ---------------------- | ---------------------------- |
| Integration route      | How is the model connected?  |
| Support level          | What does Studio maintain?   |
| Validated capabilities | What has been demonstrated?  |
| Portfolio status       | What investment is approved? |

This separates implementation choice from support and technical evidence.

---

## Support Levels

| Level                  | Claim                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| `Available`            | Accessible through a generic adapter; upstream limitations apply |
| `Studio-supported`     | Studio maintains documented model workflows                      |
| `Deployment-supported` | Studio and Runtime maintain a validated OpenVINO deployment path |

Generic availability does not create a model-specific OpenVINO commitment.

---

## Capability Evidence

Track independently:

- Studio integration
- XPU training
- XPU fine-tuning
- XPU eager inference
- Runtime/OpenVINO deployment
- Quantized inference

Do not use `XPU-enabled` without naming the validated workflow.

---

## OpenVINO Policy

OpenVINO is Intel's preferred production deployment path.

- Assess feasibility for every Studio-supported model.
- Require OpenVINO for Deployment-supported status.
- Evaluate quantization after the graph is stable.
- Keep blocked committed work owned, evidenced, and reviewed.
- Measure complete target scenarios, not tensor parity alone.

---

## Implementation Order

```text
Generic framework or repository adapter
                ↓
Named integration and supported workflows
                ↓
XPU and OpenVINO through the existing path
                ↓
First-party only when evidence justifies it
```

The architecture supports LeRobot, future frameworks, and standalone models.

---

## Cross-Team Model

| Work area | Accountable outcome                                   |
| --------- | ----------------------------------------------------- |
| Product   | End-to-end Studio and Runtime support                 |
| Platform  | Reusable XPU, OpenVINO, and quantization capabilities |
| Ecosystem | Durable support in priority external projects         |

Group blockers by root cause and affected models.

---

## Leadership Decisions

- Name an ecosystem DRI.
- Allocate maintained Intel CI for priority external projects.
- Establish recurring XPU and OpenVINO blocker triage.
- Include robot learning in platform workload planning.

LeRobot is a current high-leverage target, not the only target.

---

## Measures

Track breadth and depth separately:

- Available, Studio-supported, and Deployment-supported model counts.
- Time to Studio support and OpenVINO deployment.
- XPU capabilities validated by workflow.
- OpenVINO and quantization performance improvement.
- Shared blocker impact and downstream patch removal rate.

---

## Proposal

Adopt:

- `model-implementation-guidelines.md` as Studio/Runtime policy.
- `intel-enablement-strategy.md` as the cross-team operating proposal.
- `model-implementation-guidelines-reference.md` as the engineering checklist.

Review model commitments and shared blockers quarterly.
