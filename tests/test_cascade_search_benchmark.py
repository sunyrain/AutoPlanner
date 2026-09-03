from cascade_planner.cascade_search.proposals import StaticProposalProvider
from cascade_planner.cascade_search.search import CascadeProgramSearch, CascadeSearchConfig
from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeActionType,
    StepAnnotation,
)
from cascade_planner.eval.run_cascade_search_benchmark import (
    _merge_proposal_caches,
    build_parser,
)


def _action(rxn_smiles: str, source: str, total_cost: float) -> CascadeAction:
    reactants, product = rxn_smiles.split(">>")
    return CascadeAction(
        CascadeActionType.RETROSYNTHETIC_STEP,
        target_leaf=product,
        source=source,
        step=StepAnnotation(
            product_smiles=product,
            reactant_smiles=reactants.split("."),
            rxn_smiles=rxn_smiles,
            source_model=source,
            score=1.0 - total_cost,
            raw_metadata={"cascade_cost": {"total_cost": total_cost}},
        ),
    )


def test_proposal_cache_merge_reserves_complete_and_exploration_candidates() -> None:
    product = "CCO"
    complete = {
        product: [
            _action(f"C{i}>>{product}", "complete-route", 100.0 + i)
            for i in range(3)
        ]
    }
    exploration = {
        product: [
            _action(f"N{i}>>{product}", "expansion-trace", float(i))
            for i in range(8)
        ]
    }

    merged = _merge_proposal_caches(complete, exploration)[product]

    assert [action.source for action in merged[:6]] == [
        "complete-route",
        "expansion-trace",
        "complete-route",
        "expansion-trace",
        "complete-route",
        "expansion-trace",
    ]


def test_proposal_cache_merge_keeps_primary_copy_of_duplicate_reaction() -> None:
    product = "CCO"
    reaction = f"CC=O>>{product}"
    primary = _action(reaction, "complete-route", 100.0)
    duplicate = _action(reaction, "expansion-trace", 0.1)

    merged = _merge_proposal_caches(
        {product: [primary]},
        {product: [duplicate]},
    )[product]

    assert merged == [primary]


def test_search_can_continue_past_return_limit_to_explore_multistep_routes() -> None:
    target = "CCO"
    intermediate = "CC=O"
    provider = StaticProposalProvider(
        {
            target: [
                _action(f"C>>{target}", "direct", 0.1),
                _action(f"{intermediate}>>{target}", "multistep", 0.2),
            ],
            intermediate: [_action(f"CC>>{intermediate}", "multistep", 0.1)],
        }
    )
    def stock(smiles: str) -> bool:
        return smiles in {"C", "CC"}

    planner = CascadeProgramSearch(
        [provider],
        stock_checker=stock,
        config=CascadeSearchConfig(
            max_depth=3,
            branch_factor=2,
            expansion_budget=4,
            continue_after_result_limit=True,
        ),
    )

    results = planner.search(target, n_results=1)

    assert len(results) == 1
    assert planner.stats.solved_programs == 2
    assert planner.stats.expansions == 2
    assert planner.stats.stop_reason == "queue_exhausted"


def test_benchmark_parser_defaults_are_multistep_capable() -> None:
    args = build_parser().parse_args(["--output", "out.json"])

    assert args.iterations == 10
    assert args.use_chem_enzy_expansion_proposals is True
    assert args.cascade_leaf_beam_size == 2
    assert args.cascade_diverse_leaf_reserve == 2
