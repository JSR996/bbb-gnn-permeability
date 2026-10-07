# Architecture diagrams: current, and the three variants under test

Mermaid sources for the architectures being evaluated. The organising idea
across all four is a single distinction, established by measurement:

> `group_advantages` standardises the total reward **within the group**, so it
> is invariant to anything uniform across that group — verified for uniform
> scale, uniform shift, and a uniform multiplicative factor. Any signal that
> says "all of these are unrealistic" is therefore **erased**. Only terms
> evaluated **outside** the advantage survive: the absolute invalidity floor,
> and the KL to `pi_ref`.

That is why the size gate and `kl_coef=0.3` worked while reward reweighting,
the logit transform, the alert penalty and the Tox21 term did not.

---

## 1. Current architecture (baseline, `kl_coef=0.3`, size gate)

```mermaid
flowchart TD
    W["MLE warm start<br/>pi_theta0"] --> G["Generator policy<br/>pi_theta a_t given a_lt, z"]
    W --> REF["Reference policy pi_ref<br/>frozen copy of pi_theta0"]
    Z["z ~ N 0 I"] --> G
    G -->|"sample group of N"| CAND["Group of N candidates"]
    CAND --> SAN{"RDKit<br/>sanitization"}
    SAN -->|invalid| FLOOR["Absolute floor<br/>advantage = -2.0"]
    SAN -->|valid| GATE{"heavy atoms<br/>&gt;= 10 ?"}
    GATE -->|no| FLOOR
    GATE -->|yes| TERMS

    subgraph TERMS["Reward terms, Eq. 5"]
        D["D_phi<br/>real vs fake"]
        C["Frozen ensemble C<br/>gin + gat + gine"]
        Q["QED"]
        S["SA synthesizability"]
    end

    TERMS --> R["R = wD*D + wC*C + wQ*Q + wS*S<br/>wC anneals 0.2 to 1.5<br/>wD anneals 1.0 to 0.3"]
    R --> ADV["Group-relative advantage<br/>standardise within valid subgroup"]
    FLOOR --> ADV
    ADV --> LOSS["Clipped PPO surrogate"]
    REF -->|"KL penalty, absolute"| LOSS
    LOSS -->|"update theta"| G

    style FLOOR fill:#ffe6e6,stroke:#c00
    style REF fill:#e6f0ff,stroke:#06c
    style C fill:#fff4e6,stroke:#e80
    style ADV fill:#f0f0f0,stroke:#666
```

Red = outside the advantage (absolute). Blue = the other absolute term.
Orange = the hacked term.

---

## 2. VAE-augmented pretraining (new backbone)

Changes **only** the pretraining stage that produces `theta_0`. Rollout,
reward, advantage and GRPO are untouched: at generation time `z ~ p(z)`
exactly as before.

```mermaid
flowchart TD
    M["Real molecule m<br/>SELFIES a_1..a_T"] --> ENC["Encoder f_psi"]
    ENC --> MU["mu_psi m"]
    ENC --> SIG["sigma_psi m"]
    MU --> RP["Reparameterize<br/>z = mu + sigma * eps"]
    SIG --> RP
    EPS["eps ~ N 0 I"] --> RP
    RP --> DEC["Decoder pi_theta m given z<br/>SAME GRU as before"]

    DEC --> RECON["Reconstruction<br/>-1/T sum log pi_theta a_t"]
    MU --> KL["KL to prior, closed form"]
    SIG --> KL
    KL --> FB["Free bits<br/>max lambda, KL per dim"]
    FB --> BETA["beta k * KL<br/>annealed or cyclical"]
    RECON --> OBJ["L_pretrain = recon + beta*KL_fb"]
    BETA --> OBJ
    OBJ -->|"grad theta and psi"| DEC
    OBJ -->|"reparameterized grad"| ENC

    MU --> DIAG["Aggregate posterior<br/>Delta_mom diagnostic"]
    SIG --> DIAG
    DIAG --> TRAPB{"Delta_mom large ?<br/>Trap B"}

    DEC --> TH0["theta_0 -> pi_ref<br/>same checkpoint format"]
    TH0 --> ROLL["Rollout: z ~ p z<br/>UNCHANGED"]

    style DEC fill:#e6ffe6,stroke:#0a0
    style TRAPB fill:#ffe6e6,stroke:#c00
    style ROLL fill:#e6f0ff,stroke:#06c
```

**Measured verdict: no benefit downstream, and a small validity cost.**

The first read of this ("worse on every seed", downstream scaffold 0.770 vs
0.816 with C) is withdrawn: every GRPO seed in that design shared ONE
pretrained checkpoint per arm, so the thing being compared had n=1 while the
seeds only resampled the rollouts. Warm-start variance is ~2x GRPO-seed
variance here, so that design put the larger variance component inside the
quantity it held fixed.

Re-run with a per-seed warm start (`results/factorial_ws`, VAE minus MLE,
4 seeds, paired on seed and checkpoint):

| metric | w_C on | w_C off | seeds improved |
|---|---|---|---|
| scaffold_frac | -0.052 | -0.001 | 1/4, 1/4 |
| c_mean | +0.000 | -0.001 | 2/4, 2/4 |
| tanimoto_dist | -0.001 | +0.005 | 2/4, 4/4 |
| valid_frac | **-0.014** | **-0.016** | **0/4, 0/4** |

Diversity and permeability are noise — means at or near zero, sign agreement
at chance. The one effect that holds is validity: the VAE decoder is less
valid on every seed in both cells, by about 1.5 points absolute. So the
honest claim is not "the VAE is worse at generating molecules", it is "the
VAE buys nothing downstream and costs a little validity".

Trap A did occur, and that part stands independently of the pairing because
it is measured on the pretraining objective, not on GRPO: true latent
information is 1.07 nats across 64 dims, reconstruction gain 0.048
nats/token, and raising free bits only pins KL at the floor while `Delta_mom`
degrades (1.64 to 2.61 to 5.67). Prior-sample scaffold diversity is also
genuinely lower (0.579 vs 0.730) — the latent is near-collapsed, so sampling
`z ~ p(z)` barely moves the decoder. That explains the downstream null: a
warm start whose latent carries 1 nat is, at generation time, almost the MLE
warm start.

---

## 3. No frozen ensemble (`--drop C`)

C is still **scored and logged**, so permeability stays measurable; it simply
stops steering.

```mermaid
flowchart LR
    CAND["Group of N candidates"] --> SAN{"sanitization<br/>+ size gate"}
    SAN -->|fail| FLOOR["Absolute floor"]
    SAN -->|pass| T

    subgraph T["Reward terms"]
        D["D_phi"]
        Q["QED"]
        S["SA"]
        C["Frozen ensemble C<br/>w_C = 0"]
    end

    D --> R["R = wD*D + wQ*Q + wS*S"]
    Q --> R
    S --> R
    C -.->|"logged only,<br/>no gradient"| LOG["mean_C metric"]

    R --> ADV["Group-relative advantage"]
    FLOOR --> ADV
    ADV --> LOSS["Clipped PPO + KL to pi_ref"]
    LOSS --> G["pi_theta"]

    style C fill:#eeeeee,stroke:#999,stroke-dasharray: 4 3
    style FLOOR fill:#ffe6e6,stroke:#c00
```

**Measured verdict: better realism.** mean_C 0.878 against real BBB+ at
0.858, versus 0.926 when C steers — the overshoot *is* the hack. TPSA spread
improves 0.394 to 0.462. Permeability survives because QED/SA/D/KL already
select for drug-like structure.

---

## 4. Realism gate on D_phi (proposed)

The merge of frozen ensemble and discriminator that actually works: not a
merged **score** — which group standardisation erases — but a **gate**, which
routes through the absolute floor exactly as the size gate does.

```mermaid
flowchart TD
    CAND["Group of N candidates"] --> SAN{"RDKit<br/>sanitization"}
    SAN -->|invalid| FLOOR["Absolute floor<br/>outside the advantage"]
    SAN -->|valid| SIZE{"heavy atoms<br/>&gt;= 10 ?"}
    SIZE -->|no| FLOOR
    SIZE -->|yes| DGATE{"D_phi m &gt;= tau ?<br/>AUC real vs hacked = 0.997"}
    DGATE -->|no| FLOOR
    DGATE -->|yes| T

    subgraph T["Reward terms"]
        Q["QED"]
        S["SA"]
        C["C, logged or capped at 0.9"]
    end

    T --> R["R"]
    R --> ADV["Group-relative advantage"]
    FLOOR --> ADV
    ADV --> LOSS["Clipped PPO + KL to pi_ref"]
    LOSS --> G["pi_theta"]
    G --> CAND
    G -->|"fake samples"| DTRAIN["Train D_phi<br/>outside the PPO window"]
    REAL["Real BBBP molecules"] --> DTRAIN
    DTRAIN --> DGATE

    style DGATE fill:#e6ffe6,stroke:#0a0,stroke-width:2px
    style FLOOR fill:#ffe6e6,stroke:#c00
```

**Rationale.** D scores real BBB+ drugs 0.940 and collapsed output 0.060
(AUC 0.997) — the detector already exists and is near-perfect. It fails today
because `w_D` anneals *down* to 0.3 while within a collapsed group D is
uniformly ~0.06, so its within-group sd is 0.066 and standardisation discards
it. A gate bypasses that entirely.

**Untested, and the risk is real:** D is adversarially trained, so the gate
moves between steps and the generator may learn to satisfy it without
becoming more realistic — the same Goodhart failure as the alert penalty.
Needs a 3-seed arm study with a held-out realism instrument before it is
trusted.
