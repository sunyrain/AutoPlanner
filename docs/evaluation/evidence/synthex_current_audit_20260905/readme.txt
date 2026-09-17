<div align="center">

<img src="docs/figures/synthex_logo.svg" alt="SynthEx" width="180">

# SynthEx

**Strategy-first synthesis planning for complex natural products**

*An agentic framework that plans routes to molecules conventional retrosynthesis tools cannot reach.*

[![Paper](https://img.shields.io/badge/paper-preprint-b31b1b)](https://arxiv.org/abs/2608.07454)
[![SynthAtlas](https://img.shields.io/badge/explore-synthatlas.epfl.ch-2b6cb0)](https://synthatlas.epfl.ch)
[![License](https://img.shields.io/badge/license-Apache--2.0%20(planned)-lightgrey)](#code-release)
[![Status](https://img.shields.io/badge/code-releasing%20soon-f6ad55)](#code-release)

</div>

> **The code is not here yet.** This repository is the home of SynthEx and is being prepared
> for an open-source release under the **Apache 2.0** licence in the coming weeks.
> In the meantime, every route SynthEx designed is already browsable at
> **[synthatlas.epfl.ch](https://synthatlas.epfl.ch)**.
> ⭐ **Star** this repo for visibility — and set **Watch** to be notified when the code lands.

---

## The problem

For half a century, computer-assisted synthesis planning has been built on catalogued reactions.
Modern retrosynthesis tools now report near-complete success on benchmarks drawn from that same
patent record — and then falter the moment they leave it.

Complex natural products are exactly where they fail. Densely fused, stereochemically rich
architectures demand bespoke, inventive disconnections — ring-closing cascades, skeletal
rearrangements — that occur too rarely to form robust templates and too rarely to be recalled
by a network trained on reaction frequency. The barrier is not search budget. It is **reaction space**.

Run near-exhaustively (median ≈ 2.9 × 10⁴ node expansions per target, 30 min wall-clock, no
expansion cap), a leading template planner solves **13.8%** of a 1,000+ target natural-product
benchmark. More compute does not help: the disconnections these targets need are not in the library.

## The idea

SynthEx removes the constraint at its source. Instead of *selecting* a disconnection from a fixed
template library, a language model **writes** it — as an ordered list of atom-level graph edits we
call **ReactionJSON**.

```jsonc
// "Break the C11–C13 bond, restore the C8=C16 double bond"
[
  { "op": "break_bond",        "map_a": 11, "map_b": 13 },
  { "op": "change_bond_order", "map_a":  8, "map_b": 16, "delta": 1 }
]
```

Ten primitives (`break_bond`, `add_bond`, `change_bond_order`, `change_atom`, `add_group`,
`remove_group`, `set_explicit_h`, `invert_stereocenter`, `clear_stereocenter`, `set_bond_stereo`)
are enough to express any transformation. Applying the edits to the mapped product yields the
mapped precursors **deterministically** — no SMILES generation, no atom-mapping model, no template.

Two things follow, and both matter:

1. **The model becomes a designer, not a selector.** It can propose chemistry that is common in
   named-reaction and total-synthesis knowledge but rare in any reaction catalogue — and therefore
   invisible to tools trained on one.
2. **A route becomes an editable object.** Because a route is a text document anchored on atom maps
   (**RouteJSON**), agents can critique and repair it *surgically* — reorder steps, insert a
   protection, swap a disconnection — without re-running the search that produced it.

## How it works

<div align="center">
  <img src="docs/figures/fig1_pipeline.png" alt="The SynthEx agentic pipeline" width="880">
</div>

SynthEx plans the way a chemist does: it settles on a **strategy** before it commits to a route,
then revises the route without abandoning the strategy. Five language-model agents implement this.

| Stage | Agent | What it does |
|---|---|---|
| 1 | **StrategyGenerator** | Proposes several competing high-level strategies, each anchored on a key disconnection. Accepts chemist-supplied constraints in natural language — a required starting material, a free-text instruction. |
| 2 | **RouteBuilder** | Expands each strategy into a complete pathway, writing every disconnection as ReactionJSON. Distinguishes key steps from supporting ones. |
| 3 | **Critic** | Simulates each reaction in the forward direction and flags steps that are chemically infeasible as written (*blocking* reactions). |
| 4 | **Improvement** | Repairs the blocking steps through surgical edits that leave the strategy intact. Loops with the Critic until the route is clean. |
| 5 | **Analysis** | Scores the finished route for feasibility, identifies its key steps and principal risks. |

Under the hood, the strategic layer is a Monte-Carlo tree search whose neural template expansion
policy has been replaced by an LLM-guided one; precursors the strategic layer leaves unpurchasable
are completed by a deliberately short template search.

## What we found

<table>
<tr><td width="50%" valign="top">

**Reach — a five-fold increase where it is hardest**

| Method | Solve rate |
|---|---|
| Template planner (exhaustive) | 13.8% |
| SynthEx, strategic layer only | 20.8% |
| **SynthEx, stitched** | **67.2%** |

The same template engine that solves 13.8% alone completes 67.2% of targets once SynthEx has done
the strategic work. And the margin **widens** rather than narrows as molecules grow more complex.

</td><td width="50%" valign="top">

**Reaction space — a region catalogues do not contain**

| | SynthEx | USPTO | Template top-1 |
|---|---|---|---|
| Ring-forming steps | **16.4%** | 9.9% | 2.8% |
| C–C bond formation | **22.5%** | — | 9.7% |
| Protecting-group ops | 27.0% | — | 40.0% |

A state-of-the-art single-step model ranks SynthEx's disconnection in its top-5 for only 31.4% of
targets — and 10.9% of the ring-forming ones. Two-thirds of SynthEx's transformations are simply
absent from what a corpus-trained model proposes.

</td></tr>
</table>

<div align="center">
  <img src="docs/figures/fig3_reaction_space.png" alt="SynthEx occupies a distinct reaction space" width="880">
</div>

**Chemistry experts cannot tell SynthEx's key steps from published ones.** Ten synthetic chemists
from total-synthesis groups rated key steps — drawn from SynthEx routes and from human total
syntheses published *after* the model's training cut-off, rendered identically and stripped of any
cue to their origin — on feasibility, strategic value, elegance and overall quality. SynthEx steps
were statistically indistinguishable from the published expert steps on feasibility, elegance and
overall quality. A classifier trained on the panel's own four-axis scores recovers the source of a
step at **AUC 0.48** — chance. Strategic value is the one axis where a small, rater-dependent edge
remains with the literature.

**And the routes are more concise.** Where both find a route, SynthEx's is shorter than the
template planner's on 130 of 161 shared targets (median 7 vs 11 steps).

## Chemistry worth arguing about

<div align="center">
  <img src="docs/figures/fig2_case_studies.png" alt="Three case studies in strategic reasoning" width="880">
</div>

Three targets, chosen as a deliberate gradient in how much independent validation is available:

**Okaramine M** — *recovering an expert route published after the training cut-off.* SynthEx
reconstructs the tandem prenylation / iminium-trapping key step of an experimentally validated
synthesis it cannot have seen, correctly reasoning that TIPS protection at the indole nitrogen
lowers C3 nucleophilicity so that electrophilic addition is directed to the *unprotected* indole.
It also protects earlier than the published route — a defensible refinement, offered unprompted.

**Melonine** — *converging on a disconnection an expert group committed laboratory effort to.*
With no published route available to it, SynthEx selects the same Mannich cyclization that a
total-synthesis group had attempted and abandoned: their substrate suffered a severe conformational
clash. SynthEx's tandem aza-Cope / Pictet–Spengler generates the iminium by rearrangement instead
of condensation, replacing the offending CH₂–CH₂ linker with an HC=CH double bond — which the group
that ran the original experiments judges more likely to reach a reactive conformation than their own.
It is a live experimental proposal, and one we intend to test.

**Chanoclavine → Lysergol** — *planning from an advanced intermediate.* Given only two structures
and no route in the literature across that gap, SynthEx proposes a Hofmann–Löffler–Freytag sequence:
chlorinate, photolyse, 1,6-HAT onto an allylic methyl, then close the D ring by intramolecular
N-alkylation. It installs a handle where the chemistry needs one, through remote functionalisation
of an unactivated C–H bond — precisely the low-frequency chemistry the patent record lacks.

## SynthAtlas — the routes are open, today

<div align="center">
  <img src="docs/figures/synthatlas_screenshot.png" alt="The SynthAtlas interactive resource" width="880">

### **[→ Explore SynthAtlas](https://synthatlas.epfl.ch)**
</div>

| 1,098 | 3,243 | 33,145 |
|:--:|:--:|:--:|
| natural-product targets | synthetic strategies | fully specified reaction steps |
| from NPAtlas | mean 2.87 per target | mean 10.2 per route, all atom-mapped |

Every strategy can be inspected step by step alongside the agent's reasoning, compared against the
alternative strategies proposed for the same target, and commented on by other chemists.

Two things make this more than a dump of predictions:

- **The atom-mapping is produced by construction**, not by a post-hoc mapper — precursors are
  generated by applying explicit graph edits to a mapped product. The corpus is internally
  consistent in a way post-hoc mapped datasets are not.
- **Every target was selected as having no reported total synthesis.** Each released route is
  therefore a *dated, public prediction* about a molecule nobody has yet made. We intend to report
  concordance as syntheses of these targets appear.

We hope it becomes a shared resource for a collection of complex molecules that lack any relevant
literature design — and a body of convergent, ring-forming chemistry against which the next
generation of planners can be trained and evaluated.

## What we do *not* claim

We report reach and per-step quality, not experimental feasibility.

The Critic that flags a step, the Improvement agent that repairs it and the Analysis agent that
re-scores the result are all language models: a falling blocking rate (0.27 → 0.06 over six
iterations) demonstrates convergence against the system's own criterion, not validity in a flask.
Stereochemical outcomes have not been verified, expert review surfaced occasional selectivity errors,
and the expert comparison is conditional on a shared strategic frame — it speaks to the chemistry
SynthEx proposes once a viable strategy is found, not to how reliably it finds one.

A route a chemist judges sound on paper is a hypothesis. Closing the loop in the laboratory is the
next frontier, and the planners that follow should be judged against experimental proof.

## Code release

SynthEx will be released here under the **Apache 2.0** licence. The planned release includes:

- [ ] The SynthEx agentic planner (strategy generation, route building, critic–improvement loop, analysis)
- [ ] The ReactionJSON / RouteJSON specification and reference implementation
- [ ] Configuration files to reproduce the runs reported in the paper


**Related open source releases from the group**

- [ReactionClassifier](https://github.com/schwallergroup/ReactionClassifier) — deterministic reaction classification
- [Synthelite](https://github.com/schwallergroup/synthelite) — chemist-aligned, feasibility-aware synthesis planning with LLMs
- [Synthegy](https://github.com/schwallergroup/steer) - Chemical reasoning in LLMs unlocks strategy-aware synthesis planning and reaction mechanism elucidation

## Citation

> **Strategy-first synthesis planning for complex natural products**

> Armstrong, D.\*, Nguyen, X.-V.\*, Susanu, O., Gibberd, G., Neukomm, T. A., Strunden, T.,
> Forster, D., Delattre, M., Teh, S., Rols, C., Federice, J., Leatherwood, H., Barnes, M. L.,
> Dobbelaere, M. R., Wipf, P., Njardarson, J. T., Zhu, J., Schwaller, P.
>
> *\*These authors contributed equally.* 



```bibtex
@article{armstrong2026synthex,
  title  = {Strategy-first synthesis planning for complex natural products},
  author = {Armstrong, Daniel and Nguyen, Xuan-Vu and Susanu, Octavian and Gibberd, Gabriel
            and Neukomm, Th{\'e}o A. and Strunden, Tadd{\"a}us and Forster, Dan
            and Delattre, Morgane and Teh, Shawn and Rols, Cl{\'e}ment and Federice, John
            and Leatherwood, Hayden and Barnes, M. Lavelle and Dobbelaere, Maarten R.
            and Wipf, Peter and Njardarson, Jon T. and Zhu, Jieping and Schwaller, Philippe},
  year={2026},
  eprint={2608.07454},
  archivePrefix={arXiv},
  primaryClass={cs.MA},
  url={https://arxiv.org/abs/2608.07454} 
}
```

## Contact

Questions about the method, the resource, or a target you would like planned — open an
[issue](../../issues) or write to
[philippe.schwaller@epfl.ch](mailto:philippe.schwaller@epfl.ch).

<div align="center">

**[Laboratory of Artificial Chemical Intelligence (LIAC)](https://schwallergroup.github.io) · EPFL**
with the Laboratory of Synthesis and Natural Products (LSPN, EPFL), Ghent University,
the University of Arizona and the University of Pittsburgh.

Supported by the Swiss National Science Foundation through NCCR Catalysis (225147) and
grant 214915, AiChemist and LowDataML MSCA Doctoral Networks, Intel and Merck KGaA AWASES programme, with support from Google.org and the Google Cloud Research Credits program and through the Reaxys R&D collaboration network.

</div>
