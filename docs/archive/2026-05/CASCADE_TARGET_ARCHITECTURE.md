# AutoPlanner-Cascade Target Architecture

## Thesis

AutoPlanner-Cascade should plan catalytic cascade processes, not merely
multi-step chemoenzymatic routes.

The central question is:

```text
Can these catalytic steps form a feasible one-pot, sequential one-pot,
telescoped, or staged cascade under shared or segmented conditions?
```

## Target Pipeline

```text
Target molecule + constraints
        ↓
Retrosynthetic proposal layer
  organic / enzymatic / retrieval / template proposals
        ↓
Catalyst and enzyme evidence layer
  EC, enzyme candidate, organism, sequence, substrate similarity, literature
        ↓
Condition envelope layer
  pH, temperature, solvent, buffer, salt, metal, oxygen, redox, cofactors
        ↓
Cascade-state route-tree search
  add reaction step
  assign catalyst/enzyme
  assign condition window
  merge step into stage
  split stage
  insert buffer exchange / quench / isolation
  add cofactor regeneration
        ↓
Cascade feasibility model
  step plausibility
  catalyst/enzyme match
  condition likelihood
  pairwise compatibility
  cofactor closure
  global cascade value
  uncertainty
        ↓
Independent cascade verifier
  atom/product sanity
  EC/type sanity
  pH/T overlap
  solvent/buffer conflict
  enzyme stability
  catalyst poisoning
  cofactor/redox balance
  one-pot vs sequential feasibility
        ↓
Ranked cascade plans
  route, stage partition, shared conditions, evidence pack, risks,
  uncertainty, and recommended validation experiments
```

## State Objects Needed

### ConditionEnvelope

```text
pH_min / pH_max
temperature_min / temperature_max
solvent_class
organic_cosolvent_fraction
buffer
salt / metal
oxygen_requirement
oxidant / reductant
cofactor / cosubstrate
water_activity
```

### StagePartition

```text
stages = [[step_1, step_2], [step_3], [step_4]]
operation_type = one_pot | sequential_one_pot | telescoped | isolation
required_operations = buffer_exchange | quench | extraction | immobilization
```

### CascadeLedger

```text
cofactor_balance
redox_conflicts
pH_overlap
temperature_overlap
solvent_risks
metal_enzyme_conflicts
buffer_conflicts
reactive_intermediate_risks
inhibition_flags
evidence_level
uncertainty
```

### EvidencePack

```text
reaction evidence
enzyme/catalyst evidence
condition evidence
compatibility evidence
negative evidence
model-only assumptions
recommended validation experiment
```

## Design Rule

Condition and compatibility must become search state and scoring features. They
should not remain only post-hoc metrics like `condition_window_success_any` or
`cascade_compatibility_success_any`.
