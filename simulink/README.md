# Simulink / SimEvents Model - Rural Screening Clinic Queue Simulation

## Purpose
This model simulates a telemedicine screening clinic using **SimEvents** to
evaluate the impact of AI-assisted triage on patient throughput, waiting
times, and referral load in a low-resource rural setting.

---

## Model Architecture

```
[Entity Generator]  -->  [Queue]  -->  [AI Server]  -->  [Referral / Result Handling]
        |                    |               |                       |
   (patients arrive)   (waiting room)  (AI inference)       (grade-based routing)
```

### Block Descriptions

| Block | SimEvents Block Type | Role |
|-------|---------------------|------|
| **Entity Generator** | `Entity Generator` | Creates patient entities arriving at the clinic |
| **Queue** | `Entity Queue` | Models the waiting room; can overflow at capacity |
| **AI Server** | `Entity Server` | Represents AI inference + clinician review time |
| **Referral Handling** | `Entity Terminator` + routing logic | Routes entities by DR grade to Refer / Routine / Discharge |

---

## Key Parameters (to be set in Model Workspace or Simulink Data Dictionary)

| Parameter | Symbol | Default | Units | Notes |
|-----------|--------|---------|-------|-------|
| Patient arrival rate | `lambda` | 4 | patients/hour | Exponential inter-arrival (Poisson process) |
| AI inference service time | `mu_ai` | 3 | minutes/patient | Exponential; adjust for GPU vs CPU |
| Number of AI servers | `n_servers` | 2 | N/A | Increase to model multi-device clinics |
| Queue capacity | `Q_cap` | 20 | patients | Overflow -> redirect to next clinic |
| Referral threshold | N/A | "Moderate" and above | N/A | Severe / PDR -> urgent ophthalmology referral |
| Simulation duration | `T_sim` | 480 | minutes | 8-hour clinic day |

---

## Outputs / Statistics to Collect

- Average queue length over the simulation day
- Mean waiting time per patient (queue delay)
- Server utilisation (% busy time)
- Referral count by grade (No DR / Mild / Moderate / Severe / PDR)
- Throughput (patients screened per hour)

---

## TODO (Owner: Member 4)

1. Open **MATLAB -> Simulink -> SimEvents** and create `simulink/DRClinic.slx`.
2. Place and connect the four core blocks described above.
3. Set block parameters from the table above (use `Model Workspace` for clean parameterisation).
4. Add a `Statistics` block to each key connection to log queue length and wait times.
5. Wire outputs to a `To Workspace` block or `Simulation Data Inspector`.
6. Run a parametric sweep over `lambda` and `n_servers` to find the optimal server count for <= 10 min mean wait.
7. Document findings in `results/simevents_study.md`.

---

> **Note**: This model complements the AI pipeline (not replaces it).
> It helps clinic administrators decide staffing and device allocation,
> not diagnose patients.
