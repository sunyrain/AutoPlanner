# Strategic Disconnections

This directory stores curated strategic retrosynthesis knowledge. These records
are intentionally not ordinary reaction templates and not stock records.

## Organization

- `families`: target-family level recognition and route-policy notes.
- `anchors`: named strategic starting points or fragments, such as steroid
  chiral-pool anchors or statin side-chain common intermediates.
- `disconnections`: route-planning moves with applicability, suggested
  precursor roles, risks, evidence, and a use policy.
- `pptx_sources`: extraction metadata for local presentation sources.

## Source Files

The default query path merges every
`data/strategic_disconnections/strategic_disconnections*.json` file.

- `strategic_disconnections_v0.json`: local project seed for bufadienolides
  and statin route strategy extracted from local project context.
- `strategic_disconnections_public_seed_v1.json`: first public-literature seed
  for broad natural-product and semisynthesis families.
- `strategic_disconnections_public_expansion_v2.json`: expanded public source
  layer for additional medicinal-chemistry, heterocycle, semisynthesis,
  organofluorine, sulfur(VI), prodrug, and biosynthesis-aware families.
- `strategic_disconnections_public_expansion_v3.json`: second public expansion
  for medicinal heterocycles, multicomponent reaction scaffolds, click and
  bioorthogonal motifs, organoboron motifs, and compliance-gated diazepine/PBD
  privileged scaffolds.
- `strategic_disconnections_public_expansion_v4.json`: third public expansion
  for fused aza-heteroaromatics, nucleobase-like cores, saturated
  N-heterocycles, lactone/lactam motifs, quinone redox scaffolds, antibiotic
  semisynthesis anchors, siderophores, PROTACs, ADC linker-payloads, and
  radiopharmaceutical chelator-linkers.
- `strategic_disconnections_public_expansion_v5.json`: fourth public expansion
  for macrolide/ketolide, glycopeptide/lipopeptide, and oxazolidinone
  antibiotic semisynthesis, covalent inhibitor warheads, oligonucleotide
  conjugates, ASO/RNAi backbone manufacturing, ionizable LNP lipids, and
  medicinal metal complexes.
- `strategic_disconnections_public_expansion_v6.json`: fifth public expansion
  for glycosylated natural products, NRPS/RiPP peptide natural products,
  compliance-gated morphinan/tropane/quinolizidine alkaloid critique,
  ellagitannins, lipid mediators and endocannabinoids, iminosugars,
  macrocyclic degraders and molecular glues, and photoresponsive probes.
- `strategic_disconnections_public_expansion_v7.json`: sixth public expansion
  for marine ladder polyethers and meroterpenoids, monoterpene indole alkaloid
  branches, cyanobacterial peptide/PKS toxin critique, retinoid/secosteroid/
  bile-acid lipophiles, polyamine siderophores and acylpolyamines,
  host-guest supramolecular motifs, isotope/PET tracers, and polymer-drug
  conjugate linker boundaries.
- `strategic_disconnections_public_expansion_v8.json`: seventh public expansion
  for organosilicon/boron/phosphorus medicinal motifs, fluorinated lipophiles,
  halogenated marine natural products, sugar-nucleotide and glycoconjugate
  vaccine boundaries, cofactor-mimic redox-active scaffolds, non-ADC
  bioconjugation, biomimetic natural-product dimerization, and process
  impurity/metabolite route critique.
- `strategic_disconnections_public_expansion_v9.json`: eighth public expansion
  for stapled peptides and peptidomimetic macrocycles, sulfated
  glycosaminoglycan/heparin mimetics, saponin glycosides, strained-ring
  bioactive motifs, sulfur-rich natural products, metal chelators and
  ionophores, macrocyclic kinase or bifunctional ligands, and
  photoredox/electrochemical route handles.
- `strategic_disconnections_public_expansion_v10.json`: ninth public expansion
  for cyclic depsipeptides and lipopeptides, prenylated meroterpenoid
  tailoring, polyamine/DNA-crosslinker critique, lipidated peptide and
  glycolipid immunomodulators, enediyne/polyene chromophores,
  halogen/chalcogen bonding motifs, reversible covalent fragment probes, and
  stable-isotope metabolomics standards.
- `strategic_disconnections_public_expansion_v11.json`: tenth public expansion
  for photocleavable protecting groups and photoactive probes,
  hypoxia-activated and redox prodrugs, CNS/BBB route motifs, PNA/PMO/LNA/XNA
  oligomer backbones, BNCT/radiometal payloads, caged activity-based probes,
  polycyclic cage topology, and salt/cocrystal/polymorph process endpoints.
- `strategic_disconnections_public_expansion_v12.json`: eleventh public
  expansion for enzyme cofactor-recycling cascades, late-stage site-selective
  C-H functionalization, carbohydrate protection and glycosylation strategy,
  atropisomer/conformational-lock route policy, reactive-metabolite safety
  critique, prodrug/conjugate linker boundaries, cryo-EM and target-engagement
  probes, and continuous-flow photo/electrochemical process boundaries.
- `strategic_disconnections_public_expansion_v13.json`: twelfth public
  expansion for organoiodine and hypervalent-iodine route handles,
  organocatalytic stereocontrol, photoenzymatic cascades, artificial
  metalloenzymes, PFAS persistence/replacement critique, RNA/riboswitch ligand
  motifs, PPI hotspot mimetics, and solid-supported or encoded-library linker
  boundaries.
- `strategic_disconnections_public_expansion_v14.json`: thirteenth public
  expansion for enzyme-aware strategic sources, including KRED/ADH chiral
  alcohols, TA/IRED/RedAm chiral amines, enzymatic halogenation,
  P450/oxygenase late C-H oxidation, glycosyltransferase glycodiversification,
  BVMO oxygen insertion, aldolase/transketolase C-C formation, and terpene
  cyclase or PKS/NRPS biosynthetic assembly-line boundaries.
- `strategic_disconnections_public_expansion_v15.json`: fourteenth public
  expansion for additional enzyme/process strategic sources, including
  lipase/esterase resolution, nitrile-metabolizing enzymes, ene-reductases,
  methyltransferase/acyltransferase tailoring enzymes, enzymatic C1
  fixation/removal, oxygenase/peroxygenase oxygenation, chemoenzymatic DKR,
  and immobilized-enzyme or flow-biocatalysis process boundaries.
- `strategic_disconnections_public_expansion_v16.json`: fifteenth public
  expansion for source-reliability and process-boundary enzyme strategy,
  including enzyme promiscuity, whole-cell biotransformation, cell-free pathway
  reconstitution, cofactor regeneration, biocatalyst stability, product
  inhibition and in situ removal, screening lineage, and green-chemistry/LCA
  critique.
- `strategic_disconnections_public_expansion_v17.json`: sixteenth public
  expansion for reaction-source provenance and evidence-quality strategy,
  including structured reaction database schema, patent/literature extraction,
  negative or failed-reaction data, procedure metadata, atom mapping and role
  assignment, natural-product structure revision, process starting-material and
  impurity-control boundaries, and CASP benchmark lineage.
- `strategic_disconnections_public_expansion_v18.json`: seventeenth public
  expansion for route-choice boundaries beyond ordinary templates, including
  stereochemical model transfer, protecting-group orthogonality and
  deprotection sequencing, salt resolution and chiral crystallization,
  solid-phase impurity lineage, late-stage diversification and SAR lineage,
  hazardous-reagent substitution, feedstock/supply-chain risk, and analytical
  identity/reference-standard/stability policy.
- `strategic_disconnections_public_expansion_v19.json`: eighteenth public
  expansion for whole-route and source-governance boundaries, including route
  economics and PMI, scalable purification/isolation bottlenecks, formulation
  and solid-form developability, photostability/autoxidation storage liability,
  biosynthetic-origin and isotope/biogenic annotation, bioassay and binding-mode
  source validation, genotoxic/nitrosamine impurity precursor liability, and
  controlled/stewardship-sensitive/dual-use governance policy.
- `strategic_disconnections_public_expansion_v20.json`: nineteenth public
  expansion for source-decision and validation boundaries, including route
  comparison decision traces, literature/patent/database conflict resolution,
  scale-up and technology-transfer boundaries, computational prediction
  uncertainty, route robustness and design-of-experiments policy, analogue-series
  and template leakage, local project route-card manual validation, and
  strategic failure-case source records.

Current merged coverage: 178 families, 181 anchors, and 354 strategic
disconnections.

`AUDIT_SUMMARY.md` records the latest local audit snapshot. `NEXT_SOURCE_BACKLOG.md`
tracks candidate families and quality gates for the next expansion batch.

## Use Policy

Default mode is advisory only. A strategic entry can be used to:

- explain why a route is or is not meaningful,
- seed a proposal source after manual review,
- add a reranking feature,
- audit product-like terminal artifacts.

It should not automatically mark an advanced precursor as stock. That mistake
is exactly what happened in the bufotalin/deacetylbufotalin probe: a product-like
bufadienolide analogue was treated as stock and the route closed before any real
skeleton strategy was proposed.

For a new user-provided SMILES with no target-specific local data, use the
agentic blackboard workflow:

1. parse the SMILES and identify scaffold/family/candidate strategic bonds;
2. let ordinary ChemEnzy/template planning handle routine steps first;
3. when the route reaches an unresolved advanced natural-product frontier,
   search literature for the family-level key construction;
4. instantiate the strategy as separate candidate kinds:
   `exact_fragment_retro`, `forward_surrogate`, and `route_anchor`;
5. validate all non-empty SMILES/rxn SMILES and render a hybrid route figure.

Detailed runbook:

- `docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md`

Do not collapse these candidate kinds into one solved route. A
`forward_surrogate` is planning material, not a lab procedure. A `route_anchor`
is a reviewed upstream hypothesis, not ordinary stock.

## Current Seed Families

- `bufadienolide_steroid`: C17-pyrone installation, C14/C16 oxygenation, and
  steroid chiral-pool anchors.
- `synthetic_statin`: convergent synthetic statin assembly around the syn-3,5-
  dihydroxy acid side chain and heteroaryl/aromatic cores.
- `natural_statin_semisynthesis`: fermentation-derived natural statin cores and
  late-stage semisynthesis.
- `macrocycle_polyketide`: macrolactonization, RCM, and stereodefined aldol
  fragment logic.
- `indole_alkaloid` and `benzylisoquinoline_alkaloid`: Pictet-Spengler and
  biosynthetic alkaloid anchors.
- `glycoside_nucleoside`: aglycone/sugar and base/sugar disconnections.
- `peptide_beta_lactam`, `taxane_semisynthesis`, `terpene_isoprenoid`,
  `artemisinin_sesquiterpene_peroxide`, `biaryl_atropisomer`, and
  `prostaglandin_eicosanoid`: public seed families for route critique and
  proposal-source development.
- `spiroketal_polyether`, `flavonoid_chalcone_chromone`,
  `coumarin_chromenone`, `lignan_neolignan`, `steroid_semisynthesis`,
  `porphyrin_tetrapyrrole`, `quinolone_fluoroquinolone`,
  `sulfonamide_sulfone_medicinal`, `nucleotide_protide_prodrug`,
  `organofluorine_late_stage`, `c_glycoside_sglt2`,
  `cannabinoid_meroterpenoid`, `tropane_alkaloid`, and
  `xanthone_anthraquinone_polyketide`: v2 public expansion families. The
  cannabinoid and tropane entries are compliance-gated, high-level structural
  priors and must not be promoted into operational proposal generators without
  project approval.
- `triazole_click_bioorthogonal`,
  `benzimidazole_benzoxazole_benzothiazole`,
  `pyrazole_isoxazole_azole`, `benzofuran_indole_annulation`,
  `quinazoline_kinase_scaffold`, `mcr_ugi_passerini_strecker`,
  `hantzsch_biginelli_mcr_heterocycles`,
  `isoindolinone_phthalimide_imide`, `oxindole_spirooxindole_isatin`,
  `organoboron_medicinal_boronate`,
  `benzodiazepine_diazepine_privileged`, and
  `benzofused_lactam_lactone_scaffolds`: v3 public expansion families. The
  diazepine/PBD, cyanide-equivalent Strecker, azide, hydrazine, and
  bioorthogonal entries are advisory source records and require safety/legal
  review before any proposal-source promotion.
- `azaindole_indazole_imidazopyridine`, `purine_pyrimidine_heterocycle`,
  `saturated_n_heterocycle_scaffold`, `lactone_lactam_beta_lactone`,
  `quinone_hydroquinone_redox_scaffold`, `aminoglycoside_semisynthesis`,
  `tetracycline_polyketide_antibiotic`, `siderophore_hydroxamate_catechol`,
  `protac_degrader_linker`, `adc_payload_linker`, and
  `radiopharmaceutical_chelator`: v4 public expansion families. Antibiotic,
  siderophore, PROTAC, ADC, radiopharmaceutical, quinone, beta-lactone,
  hydrazine, and azide-adjacent entries are advisory source records and require
  safety, stewardship, radiochemical, biosecurity, or legal review before any
  proposal-source promotion.
- `macrolide_ketolide_antibiotic`,
  `glycopeptide_lipopeptide_antibiotic`,
  `oxazolidinone_antibacterial_scaffold`,
  `covalent_warhead_electrophile`, `peptide_oligonucleotide_conjugate`,
  `rnai_antisense_phosphorothioate`, `lipid_nanoparticle_ionizable_lipid`,
  and `metal_complex_medicinal_scaffold`: v5 public expansion families.
  Antibiotic, covalent-warhead, oligonucleotide, LNP, and metal-complex records
  are advisory source records and require antimicrobial-stewardship,
  covalent-inhibitor safety, oligonucleotide manufacturing,
  formulation/delivery, or metal-complex toxicology review before promotion.
- `natural_product_glycosylated_macrolide`,
  `nonribosomal_peptide_natural_product`,
  `alkaloid_morphinan_tropane_expansion`,
  `polyphenol_tannin_ellagitannin`,
  `lipid_mediator_endocannabinoid`,
  `carbohydrate_mimetic_iminosugar`,
  `macrocyclic_degrader_and_molecular_glue`, and
  `photopharmacology_photoaffinity_probe`: v6 public expansion families.
  Controlled/toxic alkaloid, antimicrobial/bioactive peptide, cannabinoid
  lipid-mediator, degrader, photoaffinity, and photopharmacology records are
  advisory source records and require legal, safety, stewardship, toxicology,
  photochemistry, or biological-mode review before promotion.
- `marine_polyether_ladder_and_prenylated_meroterpenoid`,
  `alkaloid_monoterpene_indole_expansion`,
  `cyanobacterial_peptide_polyketide_toxin_critique`,
  `retinoid_steroid_hormone_like_lipophile`,
  `natural_product_polyamine_siderophore_expansion`,
  `macrocyclic_host_guest_supramolecular_drug`,
  `isotope_labeled_tracer_and_pet_probe`, and
  `polymer_drug_conjugate_and_material_linker`: v7 public expansion families.
  Marine toxin, psychoactive/toxic alkaloid, cyanotoxin, cytotoxic payload,
  endocrine-active lipophile, antimicrobial-vector, radiochemical,
  host-guest formulation, and polymer-conjugate records are advisory source
  records and require explicit safety, legal, toxicology, radiochemical,
  stewardship, formulation, or biological-mode review before promotion.
- `organosilicon_germanium_boron_phosphorus_medicinal`,
  `fluorinated_macrocycle_and_trifluoromethyl_lipophile`,
  `natural_product_halogenated_marine_terpenoid`,
  `sugar_nucleotide_glycoconjugate_vaccine`,
  `cofactor_mimic_redox_active_scaffold`,
  `covalent_bioconjugation_payload_nonadc`,
  `natural_product_dimer_oligomer_biomimetic_coupling`, and
  `process_impurity_metabolite_route_critique`: v8 public expansion families.
  Metalloid, fluorinated-lipophile, marine halometabolite, vaccine/glycan,
  redox-active, bioconjugation, natural-product dimer, and impurity/metabolite
  records are advisory source records and require toxicology, regulatory,
  formulation, biological-mode, manufacturing, or process-control review before
  promotion.
- `stapled_peptide_peptidomimetic_macrocycle`,
  `glycosaminoglycan_sulfated_carbohydrate_mimetic`,
  `saponin_triterpenoid_steroidal_glycoside`,
  `strained_ring_bioactive_epoxide_aziridine`,
  `sulfur_natural_product_isothiocyanate_thioether`,
  `metal_chelator_antidote_and_ionophore`,
  `macrocyclic_kinase_inhibitor_and_bifunctional_ligand`, and
  `photoredox_electrochemical_route_handle`: v9 public expansion families.
  Bioactive peptide, anticoagulant-like glycan, vaccine-adjuvant, strained
  electrophile, sulfur bioactivation, metal-homeostasis, bioactive kinase or
  bifunctional-ligand, radical, photochemical, and electrochemical records are
  advisory source records and require biological-mode, toxicology, immunology,
  stewardship, metal-homeostasis, photochemical, electrochemical, or
  radical-safety review before promotion.
- `depsipeptide_lipopeptide_cyclic_ester_peptide`,
  `prenylated_terpene_meroterpenoid_tailoring`,
  `polyamine_alkylator_crosslinker_cytotoxic_scaffold`,
  `lipidated_peptide_glycolipid_immunomodulator`,
  `natural_product_polyene_ene_yne_chromophore`,
  `halogen_bonding_and_chalcogen_bonding_medicinal_motif`,
  `fragment_linking_covalent_reversible_probe`, and
  `stable_isotope_metabolomics_internal_standard`: v10 public expansion
  families. Cytotoxic, genotoxic, immunomodulatory, antimicrobial,
  covalent-probe, chalcogen/halogen, chromophore, and isotope-label records are
  advisory source records and require safety, biological-mode, toxicology,
  immunology, analytical-method, or element-specific review before promotion.
- `photocleavable_photoactive_protecting_group_probe`,
  `redox_prodrug_hypoxia_activated_motif`,
  `cns_privileged_scaffold_transporter_motif`,
  `peptide_nucleic_acid_morpholino_xeno_oligomer`,
  `boron_neutron_capture_and_metal_radiotherapeutic`,
  `caged_covalent_activity_based_probe`,
  `natural_product_polycyclic_cage_scaffold`, and
  `process_crystallization_salt_polymorph_route_control`: v11 public
  expansion families. Photochemical, hypoxia-activated, CNS-active,
  oligonucleotide, radiochemical, covalent-probe, bioactive cage-natural-product,
  and solid-form process records are advisory source records and require
  safety, biological-mode, radiochemical, regulatory, process, or manufacturing
  review before promotion.
- `enzyme_cascade_redox_cofactor_recycling`,
  `site_selective_c_h_borylation_halogenation_late_stage`,
  `carbohydrate_protecting_group_regioselective_glycosylation`,
  `atropisomeric_chiral_axis_and_conformational_lock`,
  `drug_metabolite_reactive_intermediate_safety_critique`,
  `prodrug_conjugate_cleavable_linker_biologics_boundary`,
  `cryo_em_fragment_probe_and_target_engagement_tag`, and
  `continuous_flow_photochemical_electrochemical_process_boundary`: v12 public
  expansion families. Biocatalytic, late-stage-functionalization, glycan,
  atropisomeric, reactive-metabolite, conjugate-linker, target-engagement-probe,
  and flow-process records are advisory source records and require chemical,
  biological, toxicology, validation, or process review before promotion.
- `organoiodine_hypervalent_iodine_route_handles`,
  `organocatalytic_asymmetric_iminium_enamine_phase_transfer`,
  `photocatalytic_biocatalytic_hybrid_cascade`,
  `metalloenzyme_artificial_enzyme_biocatalyst_design`,
  `perfluoroalkyl_pfas_degradation_and_replacement_critique`,
  `covalent_rna_targeting_and_riboswitch_ligand_motif`,
  `protein_protein_interaction_hotspot_mimetic`, and
  `solid_supported_resin_linker_parallel_synthesis_boundary`: v13 public
  expansion families. Organoiodine, organocatalytic, photobiocatalytic,
  artificial-enzyme, PFAS, RNA-targeting, PPI-mimetic, and encoded-library
  records are advisory source records and require chemical, biological,
  environmental, validation, or library-provenance review before promotion.
- `biocatalytic_ketoreductase_chiral_alcohol`,
  `biocatalytic_transaminase_imine_reductase_chiral_amine`,
  `enzymatic_halogenase_site_selective_halogenation`,
  `p450_oxygenase_late_stage_c_h_oxidation`,
  `glycosyltransferase_glycodiversification_route`,
  `baeyer_villiger_monooxygenase_lactone_lactam`,
  `aldolase_transketolase_asymmetric_c_c_bond`, and
  `terpene_cyclase_pks_nrps_biosynthetic_boundary`: v14 public expansion
  families. These enzyme-aware records are advisory source records and require
  substrate-scope, enzyme-identity, stereochemical or site-selectivity,
  cofactor/donor, pathway-context, biological-mode, and process-safety review
  before promotion.
- `lipase_esterase_resolution_and_acylation_route`,
  `nitrilase_nitrile_hydratase_amidase_route`,
  `ene_reductase_asymmetric_alkene_reduction`,
  `methyltransferase_acyltransferase_tailoring_enzyme`,
  `decarboxylase_carboxylase_c1_fixation_route`,
  `enzymatic_epoxidation_dihydroxylation_oxygenation_route`,
  `chemoenzymatic_dynamic_cascade_resolution`, and
  `enzyme_immobilization_flow_biocatalysis_process_boundary`: v15 public
  expansion families. These enzyme/process records are advisory source records
  and require substrate-scope, selectivity, endpoint, donor/cofactor,
  process-boundary, catalyst-support, lifetime, safety, and process-control
  review before promotion.
- `enzyme_promiscuity_substrate_engineering_policy`,
  `whole_cell_fermentation_biotransformation_boundary`,
  `cell_free_enzyme_cascade_pathway_reconstitution`,
  `enzyme_cofactor_regeneration_electrochemical_photochemical`,
  `biocatalyst_stability_solvent_high_substrate_load_process`,
  `enzyme_product_inhibition_in_situ_removal_boundary`,
  `biosensor_high_throughput_enzyme_screening_lineage`, and
  `green_chemistry_lca_biocatalysis_route_critique`: v16 public expansion
  families. These enzyme source-reliability and process-boundary records are
  advisory source records and require substrate-scope, assay-lineage,
  host/cell-free context, cofactor balance, stability, product-removal,
  screening-validation, sustainability, and comparator review before promotion.
- `structured_reaction_database_schema_provenance`,
  `patent_literature_reaction_extraction_boundary`,
  `negative_failed_reaction_data_policy`,
  `procedure_condition_metadata_completeness_policy`,
  `atom_mapping_role_assignment_reaction_center_policy`,
  `natural_product_structure_revision_stereochemical_evidence`,
  `process_starting_material_impurity_control_boundary`, and
  `route_benchmark_gold_trace_lineage_policy`: v17 public expansion families.
  These source-provenance and evidence-quality records are advisory source
  records and require schema/provenance, extraction confidence, negative-data
  comparability, procedure completeness, atom-map/role-lineage,
  structure-assignment, process-control, benchmark-lineage, legal, regulatory,
  or reproducibility review before promotion.
- `stereochemical_model_transfer_and_chiral_catalyst_scope`,
  `protecting_group_orthogonality_deprotection_sequence_policy`,
  `salt_resolution_crystallization_chiral_purity_boundary`,
  `solid_phase_synthesis_deletion_sequence_impurity_policy`,
  `late_stage_diversification_library_sar_source_lineage`,
  `hazardous_reagent_substitution_process_safety_policy`,
  `feedstock_supply_chain_route_risk_boundary`, and
  `analytical_identity_reference_standard_stability_policy`: v18 public
  expansion families. These route-choice and source-quality records are
  advisory source records and require stereochemical model, protecting-group
  sequence, chiral-purity, impurity-lineage, SAR/assay, process-safety,
  supply-chain, analytical-validation, storage-stability, or regulatory review
  before promotion.
- `route_economics_pmi_step_count_process_intensity_policy`,
  `scalable_purification_isolation_bottleneck_policy`,
  `formulation_salt_coformer_delivery_boundary`,
  `photostability_autoxidation_storage_liability_policy`,
  `biosynthetic_origin_isotopic_or_biogenic_annotation_policy`,
  `bioassay_binding_mode_target_validation_source_policy`,
  `impurity_genotoxic_nitrosamine_precursor_liability_policy`, and
  `regulatory_controlled_dual_use_or_stewardship_boundary`: v19 public
  expansion families. These whole-route and source-governance records are
  advisory source records and require process-economics, separation,
  formulation/developability, stability, biosynthetic-lineage,
  assay-validation, impurity-control, stewardship, dual-use, controlled-status,
  or project-governance review before promotion.
- `route_comparison_decision_trace_policy`,
  `literature_conflict_resolution_source_quality_policy`,
  `scale_up_transfer_and_tech_transfer_boundary_policy`,
  `computational_prediction_uncertainty_route_policy`,
  `route_robustness_design_of_experiments_policy`,
  `analogue_series_template_leakage_and_data_split_policy`,
  `local_project_route_card_manual_validation_policy`, and
  `strategic_failure_case_source_policy`: v20 public expansion families. These
  decision-provenance, source-quality, scale-up, model-governance, robustness,
  leakage, local-route-card, and failure-case records are advisory source
  records and require reviewer trace, source-conflict resolution,
  tech-transfer, uncertainty calibration, DoE/robustness, split-lineage,
  local-file validation, or corrected-failure evidence before promotion.

## Querying

By default, `scripts/query_strategic_disconnections.py` merges all files matching
`data/strategic_disconnections/strategic_disconnections*.json`.

```bash
python scripts/query_strategic_disconnections.py --query bufotalin
python scripts/query_strategic_disconnections.py --family macrocycle_polyketide
python scripts/query_strategic_disconnections.py --family c_glycoside_sglt2 --json
python scripts/query_strategic_disconnections.py --family triazole_click_bioorthogonal
python scripts/query_strategic_disconnections.py --family protac_degrader_linker
python scripts/query_strategic_disconnections.py --family macrolide_ketolide_antibiotic
python scripts/query_strategic_disconnections.py --family natural_product_glycosylated_macrolide
python scripts/query_strategic_disconnections.py --family photopharmacology_photoaffinity_probe
python scripts/query_strategic_disconnections.py --family marine_polyether_ladder_and_prenylated_meroterpenoid
python scripts/query_strategic_disconnections.py --family isotope_labeled_tracer_and_pet_probe
python scripts/query_strategic_disconnections.py --family organosilicon_germanium_boron_phosphorus_medicinal
python scripts/query_strategic_disconnections.py --family process_impurity_metabolite_route_critique
python scripts/query_strategic_disconnections.py --family stapled_peptide_peptidomimetic_macrocycle
python scripts/query_strategic_disconnections.py --family photoredox_electrochemical_route_handle
python scripts/query_strategic_disconnections.py --family depsipeptide_lipopeptide_cyclic_ester_peptide
python scripts/query_strategic_disconnections.py --family stable_isotope_metabolomics_internal_standard
python scripts/query_strategic_disconnections.py --family photocleavable_photoactive_protecting_group_probe
python scripts/query_strategic_disconnections.py --family process_crystallization_salt_polymorph_route_control
python scripts/query_strategic_disconnections.py --family enzyme_cascade_redox_cofactor_recycling
python scripts/query_strategic_disconnections.py --family continuous_flow_photochemical_electrochemical_process_boundary
python scripts/query_strategic_disconnections.py --family organocatalytic_asymmetric_iminium_enamine_phase_transfer
python scripts/query_strategic_disconnections.py --family solid_supported_resin_linker_parallel_synthesis_boundary
python scripts/query_strategic_disconnections.py --family biocatalytic_ketoreductase_chiral_alcohol
python scripts/query_strategic_disconnections.py --family glycosyltransferase_glycodiversification_route
python scripts/query_strategic_disconnections.py --family terpene_cyclase_pks_nrps_biosynthetic_boundary
python scripts/query_strategic_disconnections.py --family lipase_esterase_resolution_and_acylation_route
python scripts/query_strategic_disconnections.py --family ene_reductase_asymmetric_alkene_reduction
python scripts/query_strategic_disconnections.py --family enzyme_immobilization_flow_biocatalysis_process_boundary
python scripts/query_strategic_disconnections.py --family enzyme_promiscuity_substrate_engineering_policy
python scripts/query_strategic_disconnections.py --family green_chemistry_lca_biocatalysis_route_critique
python scripts/query_strategic_disconnections.py --family structured_reaction_database_schema_provenance
python scripts/query_strategic_disconnections.py --family negative_failed_reaction_data_policy
python scripts/query_strategic_disconnections.py --family process_starting_material_impurity_control_boundary
python scripts/query_strategic_disconnections.py --family route_benchmark_gold_trace_lineage_policy
python scripts/query_strategic_disconnections.py --family stereochemical_model_transfer_and_chiral_catalyst_scope
python scripts/query_strategic_disconnections.py --family salt_resolution_crystallization_chiral_purity_boundary
python scripts/query_strategic_disconnections.py --family hazardous_reagent_substitution_process_safety_policy
python scripts/query_strategic_disconnections.py --family analytical_identity_reference_standard_stability_policy
python scripts/query_strategic_disconnections.py --family route_economics_pmi_step_count_process_intensity_policy
python scripts/query_strategic_disconnections.py --family bioassay_binding_mode_target_validation_source_policy
python scripts/query_strategic_disconnections.py --family regulatory_controlled_dual_use_or_stewardship_boundary
python scripts/query_strategic_disconnections.py --family route_comparison_decision_trace_policy
python scripts/query_strategic_disconnections.py --family computational_prediction_uncertainty_route_policy
python scripts/query_strategic_disconnections.py --family strategic_failure_case_source_policy
python scripts/query_strategic_disconnections.py --query Pictet --json
python scripts/audit_strategic_disconnections.py --fail-on-issues
```

## PPTX Extraction Notes

`合作课题.pptx` contains 104 slides. Text/table content was extracted with
`python-pptx`; many chemical schemes are embedded OLE objects or images. Those
slides are recorded as evidence but should be manually/OLE-extracted before
being converted to explicit reaction SMARTS or SMILES.
