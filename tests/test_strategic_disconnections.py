import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_strategic_disconnections import audit_databases, render_markdown
from scripts.query_strategic_disconnections import load_databases, query_records


DB_GLOB = "data/strategic_disconnections/strategic_disconnections*.json"


class StrategicDisconnectionsTest(unittest.TestCase):
    def test_default_database_merge_includes_public_expansion(self):
        data = load_databases()

        self.assertGreaterEqual(len(data["families"]), 178)
        self.assertGreaterEqual(len(data["anchors"]), 181)
        self.assertGreaterEqual(len(data["disconnections"]), 354)

        family_ids = {item["family_id"] for item in data["families"]}
        self.assertIn("spiroketal_polyether", family_ids)
        self.assertIn("c_glycoside_sglt2", family_ids)
        self.assertIn("nucleotide_protide_prodrug", family_ids)
        self.assertIn("triazole_click_bioorthogonal", family_ids)
        self.assertIn("mcr_ugi_passerini_strecker", family_ids)
        self.assertIn("benzodiazepine_diazepine_privileged", family_ids)
        self.assertIn("protac_degrader_linker", family_ids)
        self.assertIn("adc_payload_linker", family_ids)
        self.assertIn("radiopharmaceutical_chelator", family_ids)
        self.assertIn("macrolide_ketolide_antibiotic", family_ids)
        self.assertIn("covalent_warhead_electrophile", family_ids)
        self.assertIn("metal_complex_medicinal_scaffold", family_ids)
        self.assertIn("natural_product_glycosylated_macrolide", family_ids)
        self.assertIn("nonribosomal_peptide_natural_product", family_ids)
        self.assertIn("alkaloid_morphinan_tropane_expansion", family_ids)
        self.assertIn("polyphenol_tannin_ellagitannin", family_ids)
        self.assertIn("lipid_mediator_endocannabinoid", family_ids)
        self.assertIn("carbohydrate_mimetic_iminosugar", family_ids)
        self.assertIn("macrocyclic_degrader_and_molecular_glue", family_ids)
        self.assertIn("photopharmacology_photoaffinity_probe", family_ids)
        self.assertIn("marine_polyether_ladder_and_prenylated_meroterpenoid", family_ids)
        self.assertIn("alkaloid_monoterpene_indole_expansion", family_ids)
        self.assertIn("cyanobacterial_peptide_polyketide_toxin_critique", family_ids)
        self.assertIn("retinoid_steroid_hormone_like_lipophile", family_ids)
        self.assertIn("natural_product_polyamine_siderophore_expansion", family_ids)
        self.assertIn("macrocyclic_host_guest_supramolecular_drug", family_ids)
        self.assertIn("isotope_labeled_tracer_and_pet_probe", family_ids)
        self.assertIn("polymer_drug_conjugate_and_material_linker", family_ids)
        self.assertIn("organosilicon_germanium_boron_phosphorus_medicinal", family_ids)
        self.assertIn("fluorinated_macrocycle_and_trifluoromethyl_lipophile", family_ids)
        self.assertIn("natural_product_halogenated_marine_terpenoid", family_ids)
        self.assertIn("sugar_nucleotide_glycoconjugate_vaccine", family_ids)
        self.assertIn("cofactor_mimic_redox_active_scaffold", family_ids)
        self.assertIn("covalent_bioconjugation_payload_nonadc", family_ids)
        self.assertIn("natural_product_dimer_oligomer_biomimetic_coupling", family_ids)
        self.assertIn("process_impurity_metabolite_route_critique", family_ids)
        self.assertIn("stapled_peptide_peptidomimetic_macrocycle", family_ids)
        self.assertIn("glycosaminoglycan_sulfated_carbohydrate_mimetic", family_ids)
        self.assertIn("saponin_triterpenoid_steroidal_glycoside", family_ids)
        self.assertIn("strained_ring_bioactive_epoxide_aziridine", family_ids)
        self.assertIn("sulfur_natural_product_isothiocyanate_thioether", family_ids)
        self.assertIn("metal_chelator_antidote_and_ionophore", family_ids)
        self.assertIn("macrocyclic_kinase_inhibitor_and_bifunctional_ligand", family_ids)
        self.assertIn("photoredox_electrochemical_route_handle", family_ids)
        self.assertIn("depsipeptide_lipopeptide_cyclic_ester_peptide", family_ids)
        self.assertIn("prenylated_terpene_meroterpenoid_tailoring", family_ids)
        self.assertIn("polyamine_alkylator_crosslinker_cytotoxic_scaffold", family_ids)
        self.assertIn("lipidated_peptide_glycolipid_immunomodulator", family_ids)
        self.assertIn("natural_product_polyene_ene_yne_chromophore", family_ids)
        self.assertIn("halogen_bonding_and_chalcogen_bonding_medicinal_motif", family_ids)
        self.assertIn("fragment_linking_covalent_reversible_probe", family_ids)
        self.assertIn("stable_isotope_metabolomics_internal_standard", family_ids)
        self.assertIn("photocleavable_photoactive_protecting_group_probe", family_ids)
        self.assertIn("redox_prodrug_hypoxia_activated_motif", family_ids)
        self.assertIn("cns_privileged_scaffold_transporter_motif", family_ids)
        self.assertIn("peptide_nucleic_acid_morpholino_xeno_oligomer", family_ids)
        self.assertIn("boron_neutron_capture_and_metal_radiotherapeutic", family_ids)
        self.assertIn("caged_covalent_activity_based_probe", family_ids)
        self.assertIn("natural_product_polycyclic_cage_scaffold", family_ids)
        self.assertIn("process_crystallization_salt_polymorph_route_control", family_ids)
        self.assertIn("enzyme_cascade_redox_cofactor_recycling", family_ids)
        self.assertIn("site_selective_c_h_borylation_halogenation_late_stage", family_ids)
        self.assertIn("carbohydrate_protecting_group_regioselective_glycosylation", family_ids)
        self.assertIn("atropisomeric_chiral_axis_and_conformational_lock", family_ids)
        self.assertIn("drug_metabolite_reactive_intermediate_safety_critique", family_ids)
        self.assertIn("prodrug_conjugate_cleavable_linker_biologics_boundary", family_ids)
        self.assertIn("cryo_em_fragment_probe_and_target_engagement_tag", family_ids)
        self.assertIn("continuous_flow_photochemical_electrochemical_process_boundary", family_ids)
        self.assertIn("organoiodine_hypervalent_iodine_route_handles", family_ids)
        self.assertIn("organocatalytic_asymmetric_iminium_enamine_phase_transfer", family_ids)
        self.assertIn("photocatalytic_biocatalytic_hybrid_cascade", family_ids)
        self.assertIn("metalloenzyme_artificial_enzyme_biocatalyst_design", family_ids)
        self.assertIn("perfluoroalkyl_pfas_degradation_and_replacement_critique", family_ids)
        self.assertIn("covalent_rna_targeting_and_riboswitch_ligand_motif", family_ids)
        self.assertIn("protein_protein_interaction_hotspot_mimetic", family_ids)
        self.assertIn("solid_supported_resin_linker_parallel_synthesis_boundary", family_ids)
        self.assertIn("biocatalytic_ketoreductase_chiral_alcohol", family_ids)
        self.assertIn("biocatalytic_transaminase_imine_reductase_chiral_amine", family_ids)
        self.assertIn("enzymatic_halogenase_site_selective_halogenation", family_ids)
        self.assertIn("p450_oxygenase_late_stage_c_h_oxidation", family_ids)
        self.assertIn("glycosyltransferase_glycodiversification_route", family_ids)
        self.assertIn("baeyer_villiger_monooxygenase_lactone_lactam", family_ids)
        self.assertIn("aldolase_transketolase_asymmetric_c_c_bond", family_ids)
        self.assertIn("terpene_cyclase_pks_nrps_biosynthetic_boundary", family_ids)
        self.assertIn("lipase_esterase_resolution_and_acylation_route", family_ids)
        self.assertIn("nitrilase_nitrile_hydratase_amidase_route", family_ids)
        self.assertIn("ene_reductase_asymmetric_alkene_reduction", family_ids)
        self.assertIn("methyltransferase_acyltransferase_tailoring_enzyme", family_ids)
        self.assertIn("decarboxylase_carboxylase_c1_fixation_route", family_ids)
        self.assertIn("enzymatic_epoxidation_dihydroxylation_oxygenation_route", family_ids)
        self.assertIn("chemoenzymatic_dynamic_cascade_resolution", family_ids)
        self.assertIn("enzyme_immobilization_flow_biocatalysis_process_boundary", family_ids)
        self.assertIn("enzyme_promiscuity_substrate_engineering_policy", family_ids)
        self.assertIn("whole_cell_fermentation_biotransformation_boundary", family_ids)
        self.assertIn("cell_free_enzyme_cascade_pathway_reconstitution", family_ids)
        self.assertIn("enzyme_cofactor_regeneration_electrochemical_photochemical", family_ids)
        self.assertIn("biocatalyst_stability_solvent_high_substrate_load_process", family_ids)
        self.assertIn("enzyme_product_inhibition_in_situ_removal_boundary", family_ids)
        self.assertIn("biosensor_high_throughput_enzyme_screening_lineage", family_ids)
        self.assertIn("green_chemistry_lca_biocatalysis_route_critique", family_ids)
        self.assertIn("structured_reaction_database_schema_provenance", family_ids)
        self.assertIn("patent_literature_reaction_extraction_boundary", family_ids)
        self.assertIn("negative_failed_reaction_data_policy", family_ids)
        self.assertIn("procedure_condition_metadata_completeness_policy", family_ids)
        self.assertIn("atom_mapping_role_assignment_reaction_center_policy", family_ids)
        self.assertIn("natural_product_structure_revision_stereochemical_evidence", family_ids)
        self.assertIn("process_starting_material_impurity_control_boundary", family_ids)
        self.assertIn("route_benchmark_gold_trace_lineage_policy", family_ids)
        self.assertIn("stereochemical_model_transfer_and_chiral_catalyst_scope", family_ids)
        self.assertIn("protecting_group_orthogonality_deprotection_sequence_policy", family_ids)
        self.assertIn("salt_resolution_crystallization_chiral_purity_boundary", family_ids)
        self.assertIn("solid_phase_synthesis_deletion_sequence_impurity_policy", family_ids)
        self.assertIn("late_stage_diversification_library_sar_source_lineage", family_ids)
        self.assertIn("hazardous_reagent_substitution_process_safety_policy", family_ids)
        self.assertIn("feedstock_supply_chain_route_risk_boundary", family_ids)
        self.assertIn("analytical_identity_reference_standard_stability_policy", family_ids)
        self.assertIn("route_economics_pmi_step_count_process_intensity_policy", family_ids)
        self.assertIn("scalable_purification_isolation_bottleneck_policy", family_ids)
        self.assertIn("formulation_salt_coformer_delivery_boundary", family_ids)
        self.assertIn("photostability_autoxidation_storage_liability_policy", family_ids)
        self.assertIn("biosynthetic_origin_isotopic_or_biogenic_annotation_policy", family_ids)
        self.assertIn("bioassay_binding_mode_target_validation_source_policy", family_ids)
        self.assertIn("impurity_genotoxic_nitrosamine_precursor_liability_policy", family_ids)
        self.assertIn("regulatory_controlled_dual_use_or_stewardship_boundary", family_ids)
        self.assertIn("route_comparison_decision_trace_policy", family_ids)
        self.assertIn("literature_conflict_resolution_source_quality_policy", family_ids)
        self.assertIn("scale_up_transfer_and_tech_transfer_boundary_policy", family_ids)
        self.assertIn("computational_prediction_uncertainty_route_policy", family_ids)
        self.assertIn("route_robustness_design_of_experiments_policy", family_ids)
        self.assertIn("analogue_series_template_leakage_and_data_split_policy", family_ids)
        self.assertIn("local_project_route_card_manual_validation_policy", family_ids)
        self.assertIn("strategic_failure_case_source_policy", family_ids)

    def test_all_database_ids_are_unique_and_family_coverage_is_complete(self):
        records = _load_source_records()
        key_by_section = {
            "families": "family_id",
            "anchors": "anchor_id",
            "disconnections": "id",
        }

        for section, key_name in key_by_section.items():
            ids = [item[key_name] for _, item in records[section]]
            duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
            self.assertEqual(duplicates, [], section)

        coverage = defaultdict(lambda: defaultdict(int))
        for _, family in records["families"]:
            coverage[family["family_id"]]["families"] += 1
        for _, anchor in records["anchors"]:
            coverage[anchor["family_id"]]["anchors"] += 1
        for _, disconnection in records["disconnections"]:
            coverage[disconnection["family_id"]]["disconnections"] += 1

        missing = {
            family_id: dict(counts)
            for family_id, counts in coverage.items()
            if counts["families"] != 1 or counts["anchors"] < 1 or counts["disconnections"] < 1
        }
        self.assertEqual(missing, {})

    def test_public_expansions_have_traceable_evidence_and_policy(self):
        for path in [
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v2.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v3.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v4.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v5.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v6.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v7.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v8.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v9.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v10.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v11.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v12.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v13.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v14.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v15.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v16.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v17.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v18.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v19.json"),
            Path("data/strategic_disconnections/strategic_disconnections_public_expansion_v20.json"),
        ]:
            expansion = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(expansion["use_policy"]["default"], "advisory_only")
            self.assertTrue(expansion["use_policy"]["do_not_treat_as_stock_by_default"])
            self.assertIn("safety_scope", expansion["use_policy"])

            for disconnection in expansion["disconnections"]:
                evidence = disconnection.get("evidence") or []
                self.assertGreaterEqual(len(evidence), 1, disconnection["id"])
                self.assertTrue(
                    any(str(item.get("url", "")).startswith("https://") for item in evidence),
                    disconnection["id"],
                )
                self.assertIn("use_policy", disconnection, disconnection["id"])
                self.assertIn("planner_hint", disconnection["retrosynthetic_move"], disconnection["id"])

    def test_query_finds_new_strategic_families(self):
        data = load_databases()

        spiroketal = query_records(data, query="spiroketal")
        self.assertEqual(spiroketal["counts"]["families"], 1)
        self.assertEqual(spiroketal["counts"]["anchors"], 1)
        self.assertEqual(spiroketal["counts"]["disconnections"], 2)

        c_glycoside = query_records(data, family="c_glycoside_sglt2")
        self.assertEqual(c_glycoside["counts"]["families"], 1)
        self.assertEqual(c_glycoside["counts"]["anchors"], 1)
        self.assertEqual(c_glycoside["counts"]["disconnections"], 2)
        self.assertIn("C-glycosides", c_glycoside["families"][0]["name"])

        triazole = query_records(data, family="triazole_click_bioorthogonal")
        self.assertEqual(triazole["counts"]["families"], 1)
        self.assertEqual(triazole["counts"]["anchors"], 1)
        self.assertEqual(triazole["counts"]["disconnections"], 2)

        diazepine = query_records(data, family="benzodiazepine_diazepine_privileged")
        self.assertEqual(diazepine["counts"]["families"], 1)
        self.assertEqual(diazepine["counts"]["anchors"], 1)
        self.assertEqual(diazepine["counts"]["disconnections"], 2)
        self.assertIn("Compliance-gated", diazepine["families"][0]["strategic_principle"])

        protac = query_records(data, family="protac_degrader_linker")
        self.assertEqual(protac["counts"]["families"], 1)
        self.assertEqual(protac["counts"]["anchors"], 1)
        self.assertEqual(protac["counts"]["disconnections"], 2)

        radiopharma = query_records(data, family="radiopharmaceutical_chelator")
        self.assertEqual(radiopharma["counts"]["families"], 1)
        self.assertEqual(radiopharma["counts"]["anchors"], 1)
        self.assertEqual(radiopharma["counts"]["disconnections"], 1)

        macrolide = query_records(data, family="macrolide_ketolide_antibiotic")
        self.assertEqual(macrolide["counts"]["families"], 1)
        self.assertEqual(macrolide["counts"]["anchors"], 1)
        self.assertEqual(macrolide["counts"]["disconnections"], 2)

        covalent = query_records(data, family="covalent_warhead_electrophile")
        self.assertEqual(covalent["counts"]["families"], 1)
        self.assertEqual(covalent["counts"]["anchors"], 1)
        self.assertEqual(covalent["counts"]["disconnections"], 2)

        metal_complex = query_records(data, family="metal_complex_medicinal_scaffold")
        self.assertEqual(metal_complex["counts"]["families"], 1)
        self.assertEqual(metal_complex["counts"]["anchors"], 1)
        self.assertEqual(metal_complex["counts"]["disconnections"], 2)

        glycosylated_np = query_records(data, family="natural_product_glycosylated_macrolide")
        self.assertEqual(glycosylated_np["counts"]["families"], 1)
        self.assertEqual(glycosylated_np["counts"]["anchors"], 1)
        self.assertEqual(glycosylated_np["counts"]["disconnections"], 2)

        nrps = query_records(data, family="nonribosomal_peptide_natural_product")
        self.assertEqual(nrps["counts"]["families"], 1)
        self.assertEqual(nrps["counts"]["anchors"], 1)
        self.assertEqual(nrps["counts"]["disconnections"], 2)

        alkaloid = query_records(data, family="alkaloid_morphinan_tropane_expansion")
        self.assertEqual(alkaloid["counts"]["families"], 1)
        self.assertEqual(alkaloid["counts"]["anchors"], 1)
        self.assertEqual(alkaloid["counts"]["disconnections"], 3)
        self.assertIn("compliance-gated", alkaloid["families"][0]["strategic_principle"].lower())

        ellagitannin = query_records(data, family="polyphenol_tannin_ellagitannin")
        self.assertEqual(ellagitannin["counts"]["families"], 1)
        self.assertEqual(ellagitannin["counts"]["anchors"], 1)
        self.assertEqual(ellagitannin["counts"]["disconnections"], 2)

        lipid = query_records(data, family="lipid_mediator_endocannabinoid")
        self.assertEqual(lipid["counts"]["families"], 1)
        self.assertEqual(lipid["counts"]["anchors"], 1)
        self.assertEqual(lipid["counts"]["disconnections"], 2)

        iminosugar = query_records(data, family="carbohydrate_mimetic_iminosugar")
        self.assertEqual(iminosugar["counts"]["families"], 1)
        self.assertEqual(iminosugar["counts"]["anchors"], 1)
        self.assertEqual(iminosugar["counts"]["disconnections"], 2)

        degrader = query_records(data, family="macrocyclic_degrader_and_molecular_glue")
        self.assertEqual(degrader["counts"]["families"], 1)
        self.assertEqual(degrader["counts"]["anchors"], 1)
        self.assertEqual(degrader["counts"]["disconnections"], 2)

        photo_probe = query_records(data, family="photopharmacology_photoaffinity_probe")
        self.assertEqual(photo_probe["counts"]["families"], 1)
        self.assertEqual(photo_probe["counts"]["anchors"], 1)
        self.assertEqual(photo_probe["counts"]["disconnections"], 2)

        marine = query_records(data, family="marine_polyether_ladder_and_prenylated_meroterpenoid")
        self.assertEqual(marine["counts"]["families"], 1)
        self.assertEqual(marine["counts"]["anchors"], 1)
        self.assertEqual(marine["counts"]["disconnections"], 2)

        mia = query_records(data, family="alkaloid_monoterpene_indole_expansion")
        self.assertEqual(mia["counts"]["families"], 1)
        self.assertEqual(mia["counts"]["anchors"], 1)
        self.assertEqual(mia["counts"]["disconnections"], 2)

        cyanobacterial = query_records(data, family="cyanobacterial_peptide_polyketide_toxin_critique")
        self.assertEqual(cyanobacterial["counts"]["families"], 1)
        self.assertEqual(cyanobacterial["counts"]["anchors"], 1)
        self.assertEqual(cyanobacterial["counts"]["disconnections"], 2)

        lipophile = query_records(data, family="retinoid_steroid_hormone_like_lipophile")
        self.assertEqual(lipophile["counts"]["families"], 1)
        self.assertEqual(lipophile["counts"]["anchors"], 1)
        self.assertEqual(lipophile["counts"]["disconnections"], 2)

        polyamine = query_records(data, family="natural_product_polyamine_siderophore_expansion")
        self.assertEqual(polyamine["counts"]["families"], 1)
        self.assertEqual(polyamine["counts"]["anchors"], 1)
        self.assertEqual(polyamine["counts"]["disconnections"], 2)

        host_guest = query_records(data, family="macrocyclic_host_guest_supramolecular_drug")
        self.assertEqual(host_guest["counts"]["families"], 1)
        self.assertEqual(host_guest["counts"]["anchors"], 1)
        self.assertEqual(host_guest["counts"]["disconnections"], 2)

        tracer = query_records(data, family="isotope_labeled_tracer_and_pet_probe")
        self.assertEqual(tracer["counts"]["families"], 1)
        self.assertEqual(tracer["counts"]["anchors"], 1)
        self.assertEqual(tracer["counts"]["disconnections"], 2)

        polymer = query_records(data, family="polymer_drug_conjugate_and_material_linker")
        self.assertEqual(polymer["counts"]["families"], 1)
        self.assertEqual(polymer["counts"]["anchors"], 1)
        self.assertEqual(polymer["counts"]["disconnections"], 2)

        metalloid = query_records(data, family="organosilicon_germanium_boron_phosphorus_medicinal")
        self.assertEqual(metalloid["counts"]["families"], 1)
        self.assertEqual(metalloid["counts"]["anchors"], 1)
        self.assertEqual(metalloid["counts"]["disconnections"], 2)

        fluorinated = query_records(data, family="fluorinated_macrocycle_and_trifluoromethyl_lipophile")
        self.assertEqual(fluorinated["counts"]["families"], 1)
        self.assertEqual(fluorinated["counts"]["anchors"], 1)
        self.assertEqual(fluorinated["counts"]["disconnections"], 2)

        halogenated = query_records(data, family="natural_product_halogenated_marine_terpenoid")
        self.assertEqual(halogenated["counts"]["families"], 1)
        self.assertEqual(halogenated["counts"]["anchors"], 1)
        self.assertEqual(halogenated["counts"]["disconnections"], 2)

        glycoconjugate = query_records(data, family="sugar_nucleotide_glycoconjugate_vaccine")
        self.assertEqual(glycoconjugate["counts"]["families"], 1)
        self.assertEqual(glycoconjugate["counts"]["anchors"], 1)
        self.assertEqual(glycoconjugate["counts"]["disconnections"], 2)

        redox = query_records(data, family="cofactor_mimic_redox_active_scaffold")
        self.assertEqual(redox["counts"]["families"], 1)
        self.assertEqual(redox["counts"]["anchors"], 1)
        self.assertEqual(redox["counts"]["disconnections"], 2)

        bioconjugation = query_records(data, family="covalent_bioconjugation_payload_nonadc")
        self.assertEqual(bioconjugation["counts"]["families"], 1)
        self.assertEqual(bioconjugation["counts"]["anchors"], 1)
        self.assertEqual(bioconjugation["counts"]["disconnections"], 2)

        dimer = query_records(data, family="natural_product_dimer_oligomer_biomimetic_coupling")
        self.assertEqual(dimer["counts"]["families"], 1)
        self.assertEqual(dimer["counts"]["anchors"], 1)
        self.assertEqual(dimer["counts"]["disconnections"], 2)

        impurity = query_records(data, family="process_impurity_metabolite_route_critique")
        self.assertEqual(impurity["counts"]["families"], 1)
        self.assertEqual(impurity["counts"]["anchors"], 1)
        self.assertEqual(impurity["counts"]["disconnections"], 2)

        stapled = query_records(data, family="stapled_peptide_peptidomimetic_macrocycle")
        self.assertEqual(stapled["counts"]["families"], 1)
        self.assertEqual(stapled["counts"]["anchors"], 1)
        self.assertEqual(stapled["counts"]["disconnections"], 2)

        gag = query_records(data, family="glycosaminoglycan_sulfated_carbohydrate_mimetic")
        self.assertEqual(gag["counts"]["families"], 1)
        self.assertEqual(gag["counts"]["anchors"], 1)
        self.assertEqual(gag["counts"]["disconnections"], 2)

        saponin = query_records(data, family="saponin_triterpenoid_steroidal_glycoside")
        self.assertEqual(saponin["counts"]["families"], 1)
        self.assertEqual(saponin["counts"]["anchors"], 1)
        self.assertEqual(saponin["counts"]["disconnections"], 2)

        strained = query_records(data, family="strained_ring_bioactive_epoxide_aziridine")
        self.assertEqual(strained["counts"]["families"], 1)
        self.assertEqual(strained["counts"]["anchors"], 1)
        self.assertEqual(strained["counts"]["disconnections"], 2)

        sulfur = query_records(data, family="sulfur_natural_product_isothiocyanate_thioether")
        self.assertEqual(sulfur["counts"]["families"], 1)
        self.assertEqual(sulfur["counts"]["anchors"], 1)
        self.assertEqual(sulfur["counts"]["disconnections"], 2)

        chelator = query_records(data, family="metal_chelator_antidote_and_ionophore")
        self.assertEqual(chelator["counts"]["families"], 1)
        self.assertEqual(chelator["counts"]["anchors"], 1)
        self.assertEqual(chelator["counts"]["disconnections"], 2)

        macrocyclic_kinase = query_records(data, family="macrocyclic_kinase_inhibitor_and_bifunctional_ligand")
        self.assertEqual(macrocyclic_kinase["counts"]["families"], 1)
        self.assertEqual(macrocyclic_kinase["counts"]["anchors"], 1)
        self.assertEqual(macrocyclic_kinase["counts"]["disconnections"], 2)

        photoredox = query_records(data, family="photoredox_electrochemical_route_handle")
        self.assertEqual(photoredox["counts"]["families"], 1)
        self.assertEqual(photoredox["counts"]["anchors"], 1)
        self.assertEqual(photoredox["counts"]["disconnections"], 2)

        depsipeptide = query_records(data, family="depsipeptide_lipopeptide_cyclic_ester_peptide")
        self.assertEqual(depsipeptide["counts"]["families"], 1)
        self.assertEqual(depsipeptide["counts"]["anchors"], 1)
        self.assertEqual(depsipeptide["counts"]["disconnections"], 2)

        prenylated = query_records(data, family="prenylated_terpene_meroterpenoid_tailoring")
        self.assertEqual(prenylated["counts"]["families"], 1)
        self.assertEqual(prenylated["counts"]["anchors"], 1)
        self.assertEqual(prenylated["counts"]["disconnections"], 2)

        crosslinker = query_records(data, family="polyamine_alkylator_crosslinker_cytotoxic_scaffold")
        self.assertEqual(crosslinker["counts"]["families"], 1)
        self.assertEqual(crosslinker["counts"]["anchors"], 1)
        self.assertEqual(crosslinker["counts"]["disconnections"], 2)

        lipidated = query_records(data, family="lipidated_peptide_glycolipid_immunomodulator")
        self.assertEqual(lipidated["counts"]["families"], 1)
        self.assertEqual(lipidated["counts"]["anchors"], 1)
        self.assertEqual(lipidated["counts"]["disconnections"], 2)

        polyene = query_records(data, family="natural_product_polyene_ene_yne_chromophore")
        self.assertEqual(polyene["counts"]["families"], 1)
        self.assertEqual(polyene["counts"]["anchors"], 1)
        self.assertEqual(polyene["counts"]["disconnections"], 2)

        halogen_chalcogen = query_records(data, family="halogen_bonding_and_chalcogen_bonding_medicinal_motif")
        self.assertEqual(halogen_chalcogen["counts"]["families"], 1)
        self.assertEqual(halogen_chalcogen["counts"]["anchors"], 1)
        self.assertEqual(halogen_chalcogen["counts"]["disconnections"], 2)

        fragment_probe = query_records(data, family="fragment_linking_covalent_reversible_probe")
        self.assertEqual(fragment_probe["counts"]["families"], 1)
        self.assertEqual(fragment_probe["counts"]["anchors"], 1)
        self.assertEqual(fragment_probe["counts"]["disconnections"], 2)

        isotope_standard = query_records(data, family="stable_isotope_metabolomics_internal_standard")
        self.assertEqual(isotope_standard["counts"]["families"], 1)
        self.assertEqual(isotope_standard["counts"]["anchors"], 1)
        self.assertEqual(isotope_standard["counts"]["disconnections"], 2)

        photocage = query_records(data, family="photocleavable_photoactive_protecting_group_probe")
        self.assertEqual(photocage["counts"]["families"], 1)
        self.assertEqual(photocage["counts"]["anchors"], 1)
        self.assertEqual(photocage["counts"]["disconnections"], 2)

        redox_prodrug = query_records(data, family="redox_prodrug_hypoxia_activated_motif")
        self.assertEqual(redox_prodrug["counts"]["families"], 1)
        self.assertEqual(redox_prodrug["counts"]["anchors"], 1)
        self.assertEqual(redox_prodrug["counts"]["disconnections"], 2)

        cns = query_records(data, family="cns_privileged_scaffold_transporter_motif")
        self.assertEqual(cns["counts"]["families"], 1)
        self.assertEqual(cns["counts"]["anchors"], 1)
        self.assertEqual(cns["counts"]["disconnections"], 2)

        xeno_oligomer = query_records(data, family="peptide_nucleic_acid_morpholino_xeno_oligomer")
        self.assertEqual(xeno_oligomer["counts"]["families"], 1)
        self.assertEqual(xeno_oligomer["counts"]["anchors"], 1)
        self.assertEqual(xeno_oligomer["counts"]["disconnections"], 2)

        radiotherapeutic = query_records(data, family="boron_neutron_capture_and_metal_radiotherapeutic")
        self.assertEqual(radiotherapeutic["counts"]["families"], 1)
        self.assertEqual(radiotherapeutic["counts"]["anchors"], 1)
        self.assertEqual(radiotherapeutic["counts"]["disconnections"], 2)

        activity_probe = query_records(data, family="caged_covalent_activity_based_probe")
        self.assertEqual(activity_probe["counts"]["families"], 1)
        self.assertEqual(activity_probe["counts"]["anchors"], 1)
        self.assertEqual(activity_probe["counts"]["disconnections"], 2)

        cage_scaffold = query_records(data, family="natural_product_polycyclic_cage_scaffold")
        self.assertEqual(cage_scaffold["counts"]["families"], 1)
        self.assertEqual(cage_scaffold["counts"]["anchors"], 1)
        self.assertEqual(cage_scaffold["counts"]["disconnections"], 2)

        solid_form = query_records(data, family="process_crystallization_salt_polymorph_route_control")
        self.assertEqual(solid_form["counts"]["families"], 1)
        self.assertEqual(solid_form["counts"]["anchors"], 1)
        self.assertEqual(solid_form["counts"]["disconnections"], 2)

        enzyme_cofactor = query_records(data, family="enzyme_cascade_redox_cofactor_recycling")
        self.assertEqual(enzyme_cofactor["counts"]["families"], 1)
        self.assertEqual(enzyme_cofactor["counts"]["anchors"], 1)
        self.assertEqual(enzyme_cofactor["counts"]["disconnections"], 2)

        site_selective = query_records(data, family="site_selective_c_h_borylation_halogenation_late_stage")
        self.assertEqual(site_selective["counts"]["families"], 1)
        self.assertEqual(site_selective["counts"]["anchors"], 1)
        self.assertEqual(site_selective["counts"]["disconnections"], 2)

        glycosylation = query_records(data, family="carbohydrate_protecting_group_regioselective_glycosylation")
        self.assertEqual(glycosylation["counts"]["families"], 1)
        self.assertEqual(glycosylation["counts"]["anchors"], 1)
        self.assertEqual(glycosylation["counts"]["disconnections"], 2)

        atropisomer = query_records(data, family="atropisomeric_chiral_axis_and_conformational_lock")
        self.assertEqual(atropisomer["counts"]["families"], 1)
        self.assertEqual(atropisomer["counts"]["anchors"], 1)
        self.assertEqual(atropisomer["counts"]["disconnections"], 2)

        metabolite = query_records(data, family="drug_metabolite_reactive_intermediate_safety_critique")
        self.assertEqual(metabolite["counts"]["families"], 1)
        self.assertEqual(metabolite["counts"]["anchors"], 1)
        self.assertEqual(metabolite["counts"]["disconnections"], 2)

        conjugate = query_records(data, family="prodrug_conjugate_cleavable_linker_biologics_boundary")
        self.assertEqual(conjugate["counts"]["families"], 1)
        self.assertEqual(conjugate["counts"]["anchors"], 1)
        self.assertEqual(conjugate["counts"]["disconnections"], 2)

        target_engagement = query_records(data, family="cryo_em_fragment_probe_and_target_engagement_tag")
        self.assertEqual(target_engagement["counts"]["families"], 1)
        self.assertEqual(target_engagement["counts"]["anchors"], 1)
        self.assertEqual(target_engagement["counts"]["disconnections"], 2)

        flow_process = query_records(data, family="continuous_flow_photochemical_electrochemical_process_boundary")
        self.assertEqual(flow_process["counts"]["families"], 1)
        self.assertEqual(flow_process["counts"]["anchors"], 1)
        self.assertEqual(flow_process["counts"]["disconnections"], 2)

        organoiodine = query_records(data, family="organoiodine_hypervalent_iodine_route_handles")
        self.assertEqual(organoiodine["counts"]["families"], 1)
        self.assertEqual(organoiodine["counts"]["anchors"], 1)
        self.assertEqual(organoiodine["counts"]["disconnections"], 2)

        organocatalytic = query_records(data, family="organocatalytic_asymmetric_iminium_enamine_phase_transfer")
        self.assertEqual(organocatalytic["counts"]["families"], 1)
        self.assertEqual(organocatalytic["counts"]["anchors"], 1)
        self.assertEqual(organocatalytic["counts"]["disconnections"], 2)

        photobiocatalytic = query_records(data, family="photocatalytic_biocatalytic_hybrid_cascade")
        self.assertEqual(photobiocatalytic["counts"]["families"], 1)
        self.assertEqual(photobiocatalytic["counts"]["anchors"], 1)
        self.assertEqual(photobiocatalytic["counts"]["disconnections"], 2)

        artificial_enzyme = query_records(data, family="metalloenzyme_artificial_enzyme_biocatalyst_design")
        self.assertEqual(artificial_enzyme["counts"]["families"], 1)
        self.assertEqual(artificial_enzyme["counts"]["anchors"], 1)
        self.assertEqual(artificial_enzyme["counts"]["disconnections"], 2)

        pfas = query_records(data, family="perfluoroalkyl_pfas_degradation_and_replacement_critique")
        self.assertEqual(pfas["counts"]["families"], 1)
        self.assertEqual(pfas["counts"]["anchors"], 1)
        self.assertEqual(pfas["counts"]["disconnections"], 2)

        rna = query_records(data, family="covalent_rna_targeting_and_riboswitch_ligand_motif")
        self.assertEqual(rna["counts"]["families"], 1)
        self.assertEqual(rna["counts"]["anchors"], 1)
        self.assertEqual(rna["counts"]["disconnections"], 2)

        ppi = query_records(data, family="protein_protein_interaction_hotspot_mimetic")
        self.assertEqual(ppi["counts"]["families"], 1)
        self.assertEqual(ppi["counts"]["anchors"], 1)
        self.assertEqual(ppi["counts"]["disconnections"], 2)

        library_boundary = query_records(data, family="solid_supported_resin_linker_parallel_synthesis_boundary")
        self.assertEqual(library_boundary["counts"]["families"], 1)
        self.assertEqual(library_boundary["counts"]["anchors"], 1)
        self.assertEqual(library_boundary["counts"]["disconnections"], 2)

        kred = query_records(data, family="biocatalytic_ketoreductase_chiral_alcohol")
        self.assertEqual(kred["counts"]["families"], 1)
        self.assertEqual(kred["counts"]["anchors"], 1)
        self.assertEqual(kred["counts"]["disconnections"], 2)

        chiral_amine = query_records(data, family="biocatalytic_transaminase_imine_reductase_chiral_amine")
        self.assertEqual(chiral_amine["counts"]["families"], 1)
        self.assertEqual(chiral_amine["counts"]["anchors"], 1)
        self.assertEqual(chiral_amine["counts"]["disconnections"], 2)

        halogenase = query_records(data, family="enzymatic_halogenase_site_selective_halogenation")
        self.assertEqual(halogenase["counts"]["families"], 1)
        self.assertEqual(halogenase["counts"]["anchors"], 1)
        self.assertEqual(halogenase["counts"]["disconnections"], 2)

        oxygenase = query_records(data, family="p450_oxygenase_late_stage_c_h_oxidation")
        self.assertEqual(oxygenase["counts"]["families"], 1)
        self.assertEqual(oxygenase["counts"]["anchors"], 1)
        self.assertEqual(oxygenase["counts"]["disconnections"], 2)

        glycodiversification = query_records(data, family="glycosyltransferase_glycodiversification_route")
        self.assertEqual(glycodiversification["counts"]["families"], 1)
        self.assertEqual(glycodiversification["counts"]["anchors"], 1)
        self.assertEqual(glycodiversification["counts"]["disconnections"], 2)

        bvmo = query_records(data, family="baeyer_villiger_monooxygenase_lactone_lactam")
        self.assertEqual(bvmo["counts"]["families"], 1)
        self.assertEqual(bvmo["counts"]["anchors"], 1)
        self.assertEqual(bvmo["counts"]["disconnections"], 2)

        enzymatic_cc = query_records(data, family="aldolase_transketolase_asymmetric_c_c_bond")
        self.assertEqual(enzymatic_cc["counts"]["families"], 1)
        self.assertEqual(enzymatic_cc["counts"]["anchors"], 1)
        self.assertEqual(enzymatic_cc["counts"]["disconnections"], 2)

        biosynthetic_boundary = query_records(data, family="terpene_cyclase_pks_nrps_biosynthetic_boundary")
        self.assertEqual(biosynthetic_boundary["counts"]["families"], 1)
        self.assertEqual(biosynthetic_boundary["counts"]["anchors"], 1)
        self.assertEqual(biosynthetic_boundary["counts"]["disconnections"], 2)

        lipase = query_records(data, family="lipase_esterase_resolution_and_acylation_route")
        self.assertEqual(lipase["counts"]["families"], 1)
        self.assertEqual(lipase["counts"]["anchors"], 1)
        self.assertEqual(lipase["counts"]["disconnections"], 2)

        nitrile = query_records(data, family="nitrilase_nitrile_hydratase_amidase_route")
        self.assertEqual(nitrile["counts"]["families"], 1)
        self.assertEqual(nitrile["counts"]["anchors"], 1)
        self.assertEqual(nitrile["counts"]["disconnections"], 2)

        ene_reductase = query_records(data, family="ene_reductase_asymmetric_alkene_reduction")
        self.assertEqual(ene_reductase["counts"]["families"], 1)
        self.assertEqual(ene_reductase["counts"]["anchors"], 1)
        self.assertEqual(ene_reductase["counts"]["disconnections"], 2)

        tailoring = query_records(data, family="methyltransferase_acyltransferase_tailoring_enzyme")
        self.assertEqual(tailoring["counts"]["families"], 1)
        self.assertEqual(tailoring["counts"]["anchors"], 1)
        self.assertEqual(tailoring["counts"]["disconnections"], 2)

        c1_fixation = query_records(data, family="decarboxylase_carboxylase_c1_fixation_route")
        self.assertEqual(c1_fixation["counts"]["families"], 1)
        self.assertEqual(c1_fixation["counts"]["anchors"], 1)
        self.assertEqual(c1_fixation["counts"]["disconnections"], 2)

        enzymatic_oxygenation = query_records(data, family="enzymatic_epoxidation_dihydroxylation_oxygenation_route")
        self.assertEqual(enzymatic_oxygenation["counts"]["families"], 1)
        self.assertEqual(enzymatic_oxygenation["counts"]["anchors"], 1)
        self.assertEqual(enzymatic_oxygenation["counts"]["disconnections"], 2)

        chemoenzymatic_dkr = query_records(data, family="chemoenzymatic_dynamic_cascade_resolution")
        self.assertEqual(chemoenzymatic_dkr["counts"]["families"], 1)
        self.assertEqual(chemoenzymatic_dkr["counts"]["anchors"], 1)
        self.assertEqual(chemoenzymatic_dkr["counts"]["disconnections"], 2)

        flow_biocatalysis = query_records(data, family="enzyme_immobilization_flow_biocatalysis_process_boundary")
        self.assertEqual(flow_biocatalysis["counts"]["families"], 1)
        self.assertEqual(flow_biocatalysis["counts"]["anchors"], 1)
        self.assertEqual(flow_biocatalysis["counts"]["disconnections"], 2)

        enzyme_promiscuity = query_records(data, family="enzyme_promiscuity_substrate_engineering_policy")
        self.assertEqual(enzyme_promiscuity["counts"]["families"], 1)
        self.assertEqual(enzyme_promiscuity["counts"]["anchors"], 1)
        self.assertEqual(enzyme_promiscuity["counts"]["disconnections"], 2)

        whole_cell = query_records(data, family="whole_cell_fermentation_biotransformation_boundary")
        self.assertEqual(whole_cell["counts"]["families"], 1)
        self.assertEqual(whole_cell["counts"]["anchors"], 1)
        self.assertEqual(whole_cell["counts"]["disconnections"], 2)

        cell_free = query_records(data, family="cell_free_enzyme_cascade_pathway_reconstitution")
        self.assertEqual(cell_free["counts"]["families"], 1)
        self.assertEqual(cell_free["counts"]["anchors"], 1)
        self.assertEqual(cell_free["counts"]["disconnections"], 2)

        cofactor_regeneration = query_records(
            data, family="enzyme_cofactor_regeneration_electrochemical_photochemical"
        )
        self.assertEqual(cofactor_regeneration["counts"]["families"], 1)
        self.assertEqual(cofactor_regeneration["counts"]["anchors"], 1)
        self.assertEqual(cofactor_regeneration["counts"]["disconnections"], 2)

        biocatalyst_stability = query_records(data, family="biocatalyst_stability_solvent_high_substrate_load_process")
        self.assertEqual(biocatalyst_stability["counts"]["families"], 1)
        self.assertEqual(biocatalyst_stability["counts"]["anchors"], 1)
        self.assertEqual(biocatalyst_stability["counts"]["disconnections"], 2)

        product_removal = query_records(data, family="enzyme_product_inhibition_in_situ_removal_boundary")
        self.assertEqual(product_removal["counts"]["families"], 1)
        self.assertEqual(product_removal["counts"]["anchors"], 1)
        self.assertEqual(product_removal["counts"]["disconnections"], 2)

        screening_lineage = query_records(data, family="biosensor_high_throughput_enzyme_screening_lineage")
        self.assertEqual(screening_lineage["counts"]["families"], 1)
        self.assertEqual(screening_lineage["counts"]["anchors"], 1)
        self.assertEqual(screening_lineage["counts"]["disconnections"], 2)

        green_chemistry = query_records(data, family="green_chemistry_lca_biocatalysis_route_critique")
        self.assertEqual(green_chemistry["counts"]["families"], 1)
        self.assertEqual(green_chemistry["counts"]["anchors"], 1)
        self.assertEqual(green_chemistry["counts"]["disconnections"], 2)

        reaction_schema = query_records(data, family="structured_reaction_database_schema_provenance")
        self.assertEqual(reaction_schema["counts"]["families"], 1)
        self.assertEqual(reaction_schema["counts"]["anchors"], 1)
        self.assertEqual(reaction_schema["counts"]["disconnections"], 2)

        extraction = query_records(data, family="patent_literature_reaction_extraction_boundary")
        self.assertEqual(extraction["counts"]["families"], 1)
        self.assertEqual(extraction["counts"]["anchors"], 1)
        self.assertEqual(extraction["counts"]["disconnections"], 2)

        negative_data = query_records(data, family="negative_failed_reaction_data_policy")
        self.assertEqual(negative_data["counts"]["families"], 1)
        self.assertEqual(negative_data["counts"]["anchors"], 1)
        self.assertEqual(negative_data["counts"]["disconnections"], 2)

        procedure_metadata = query_records(data, family="procedure_condition_metadata_completeness_policy")
        self.assertEqual(procedure_metadata["counts"]["families"], 1)
        self.assertEqual(procedure_metadata["counts"]["anchors"], 1)
        self.assertEqual(procedure_metadata["counts"]["disconnections"], 2)

        atom_mapping = query_records(data, family="atom_mapping_role_assignment_reaction_center_policy")
        self.assertEqual(atom_mapping["counts"]["families"], 1)
        self.assertEqual(atom_mapping["counts"]["anchors"], 1)
        self.assertEqual(atom_mapping["counts"]["disconnections"], 2)

        structure_revision = query_records(data, family="natural_product_structure_revision_stereochemical_evidence")
        self.assertEqual(structure_revision["counts"]["families"], 1)
        self.assertEqual(structure_revision["counts"]["anchors"], 1)
        self.assertEqual(structure_revision["counts"]["disconnections"], 2)

        process_control = query_records(data, family="process_starting_material_impurity_control_boundary")
        self.assertEqual(process_control["counts"]["families"], 1)
        self.assertEqual(process_control["counts"]["anchors"], 1)
        self.assertEqual(process_control["counts"]["disconnections"], 2)

        route_benchmark = query_records(data, family="route_benchmark_gold_trace_lineage_policy")
        self.assertEqual(route_benchmark["counts"]["families"], 1)
        self.assertEqual(route_benchmark["counts"]["anchors"], 1)
        self.assertEqual(route_benchmark["counts"]["disconnections"], 2)

        stereochemical_scope = query_records(
            data, family="stereochemical_model_transfer_and_chiral_catalyst_scope"
        )
        self.assertEqual(stereochemical_scope["counts"]["families"], 1)
        self.assertEqual(stereochemical_scope["counts"]["anchors"], 1)
        self.assertEqual(stereochemical_scope["counts"]["disconnections"], 2)

        protecting_group = query_records(
            data, family="protecting_group_orthogonality_deprotection_sequence_policy"
        )
        self.assertEqual(protecting_group["counts"]["families"], 1)
        self.assertEqual(protecting_group["counts"]["anchors"], 1)
        self.assertEqual(protecting_group["counts"]["disconnections"], 2)

        chiral_resolution = query_records(data, family="salt_resolution_crystallization_chiral_purity_boundary")
        self.assertEqual(chiral_resolution["counts"]["families"], 1)
        self.assertEqual(chiral_resolution["counts"]["anchors"], 1)
        self.assertEqual(chiral_resolution["counts"]["disconnections"], 2)

        solid_phase = query_records(data, family="solid_phase_synthesis_deletion_sequence_impurity_policy")
        self.assertEqual(solid_phase["counts"]["families"], 1)
        self.assertEqual(solid_phase["counts"]["anchors"], 1)
        self.assertEqual(solid_phase["counts"]["disconnections"], 2)

        sar_lineage = query_records(data, family="late_stage_diversification_library_sar_source_lineage")
        self.assertEqual(sar_lineage["counts"]["families"], 1)
        self.assertEqual(sar_lineage["counts"]["anchors"], 1)
        self.assertEqual(sar_lineage["counts"]["disconnections"], 2)

        hazard_substitution = query_records(data, family="hazardous_reagent_substitution_process_safety_policy")
        self.assertEqual(hazard_substitution["counts"]["families"], 1)
        self.assertEqual(hazard_substitution["counts"]["anchors"], 1)
        self.assertEqual(hazard_substitution["counts"]["disconnections"], 2)

        supply_chain = query_records(data, family="feedstock_supply_chain_route_risk_boundary")
        self.assertEqual(supply_chain["counts"]["families"], 1)
        self.assertEqual(supply_chain["counts"]["anchors"], 1)
        self.assertEqual(supply_chain["counts"]["disconnections"], 2)

        analytical_identity = query_records(data, family="analytical_identity_reference_standard_stability_policy")
        self.assertEqual(analytical_identity["counts"]["families"], 1)
        self.assertEqual(analytical_identity["counts"]["anchors"], 1)
        self.assertEqual(analytical_identity["counts"]["disconnections"], 2)

        route_economics = query_records(
            data, family="route_economics_pmi_step_count_process_intensity_policy"
        )
        self.assertEqual(route_economics["counts"]["families"], 1)
        self.assertEqual(route_economics["counts"]["anchors"], 1)
        self.assertEqual(route_economics["counts"]["disconnections"], 2)

        purification = query_records(data, family="scalable_purification_isolation_bottleneck_policy")
        self.assertEqual(purification["counts"]["families"], 1)
        self.assertEqual(purification["counts"]["anchors"], 1)
        self.assertEqual(purification["counts"]["disconnections"], 2)

        formulation = query_records(data, family="formulation_salt_coformer_delivery_boundary")
        self.assertEqual(formulation["counts"]["families"], 1)
        self.assertEqual(formulation["counts"]["anchors"], 1)
        self.assertEqual(formulation["counts"]["disconnections"], 2)

        stability = query_records(data, family="photostability_autoxidation_storage_liability_policy")
        self.assertEqual(stability["counts"]["families"], 1)
        self.assertEqual(stability["counts"]["anchors"], 1)
        self.assertEqual(stability["counts"]["disconnections"], 2)

        biosynthetic_origin = query_records(
            data, family="biosynthetic_origin_isotopic_or_biogenic_annotation_policy"
        )
        self.assertEqual(biosynthetic_origin["counts"]["families"], 1)
        self.assertEqual(biosynthetic_origin["counts"]["anchors"], 1)
        self.assertEqual(biosynthetic_origin["counts"]["disconnections"], 2)

        bioassay = query_records(data, family="bioassay_binding_mode_target_validation_source_policy")
        self.assertEqual(bioassay["counts"]["families"], 1)
        self.assertEqual(bioassay["counts"]["anchors"], 1)
        self.assertEqual(bioassay["counts"]["disconnections"], 2)

        impurity_liability = query_records(
            data, family="impurity_genotoxic_nitrosamine_precursor_liability_policy"
        )
        self.assertEqual(impurity_liability["counts"]["families"], 1)
        self.assertEqual(impurity_liability["counts"]["anchors"], 1)
        self.assertEqual(impurity_liability["counts"]["disconnections"], 2)

        governance = query_records(data, family="regulatory_controlled_dual_use_or_stewardship_boundary")
        self.assertEqual(governance["counts"]["families"], 1)
        self.assertEqual(governance["counts"]["anchors"], 1)
        self.assertEqual(governance["counts"]["disconnections"], 2)

        route_decision = query_records(data, family="route_comparison_decision_trace_policy")
        self.assertEqual(route_decision["counts"]["families"], 1)
        self.assertEqual(route_decision["counts"]["anchors"], 1)
        self.assertEqual(route_decision["counts"]["disconnections"], 2)

        source_conflict = query_records(data, family="literature_conflict_resolution_source_quality_policy")
        self.assertEqual(source_conflict["counts"]["families"], 1)
        self.assertEqual(source_conflict["counts"]["anchors"], 1)
        self.assertEqual(source_conflict["counts"]["disconnections"], 2)

        scale_up = query_records(data, family="scale_up_transfer_and_tech_transfer_boundary_policy")
        self.assertEqual(scale_up["counts"]["families"], 1)
        self.assertEqual(scale_up["counts"]["anchors"], 1)
        self.assertEqual(scale_up["counts"]["disconnections"], 2)

        model_uncertainty = query_records(data, family="computational_prediction_uncertainty_route_policy")
        self.assertEqual(model_uncertainty["counts"]["families"], 1)
        self.assertEqual(model_uncertainty["counts"]["anchors"], 1)
        self.assertEqual(model_uncertainty["counts"]["disconnections"], 2)

        robustness = query_records(data, family="route_robustness_design_of_experiments_policy")
        self.assertEqual(robustness["counts"]["families"], 1)
        self.assertEqual(robustness["counts"]["anchors"], 1)
        self.assertEqual(robustness["counts"]["disconnections"], 2)

        leakage = query_records(data, family="analogue_series_template_leakage_and_data_split_policy")
        self.assertEqual(leakage["counts"]["families"], 1)
        self.assertEqual(leakage["counts"]["anchors"], 1)
        self.assertEqual(leakage["counts"]["disconnections"], 2)

        local_route_card = query_records(data, family="local_project_route_card_manual_validation_policy")
        self.assertEqual(local_route_card["counts"]["families"], 1)
        self.assertEqual(local_route_card["counts"]["anchors"], 1)
        self.assertEqual(local_route_card["counts"]["disconnections"], 2)

        failure_case = query_records(data, family="strategic_failure_case_source_policy")
        self.assertEqual(failure_case["counts"]["families"], 1)
        self.assertEqual(failure_case["counts"]["anchors"], 1)
        self.assertEqual(failure_case["counts"]["disconnections"], 2)

    def test_audit_reports_current_database_as_passing(self):
        report = audit_databases()

        self.assertTrue(report["passed"])
        self.assertEqual(report["issue_count"], 0)
        self.assertEqual(report["totals"]["families"], 186)
        self.assertEqual(report["totals"]["anchors"], 189)
        self.assertEqual(report["totals"]["disconnections"], 371)
        self.assertEqual(report["coverage_issues"], [])
        self.assertEqual(report["evidence"]["disconnections_without_evidence"], [])
        self.assertGreaterEqual(report["evidence"]["local_refs"], 1)
        self.assertEqual(report["evidence"]["evidence_items_without_trace"], [])
        self.assertGreaterEqual(len(report["policy"]["compliance_gated_disconnections"]), 263)

        markdown = render_markdown(report)
        self.assertIn("# Strategic Disconnection Source Audit", markdown)
        self.assertIn("Families: `186`", markdown)
        self.assertIn("Compliance-gated disconnections", markdown)

    def test_audit_detects_duplicate_ids_and_missing_traceable_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "strategic_disconnections_a.json"
            second = root / "strategic_disconnections_b.json"
            first.write_text(
                json.dumps(
                    {
                        "schema_version": "unit.a",
                        "families": [{"family_id": "family_a", "name": "Family A"}],
                        "anchors": [{"anchor_id": "anchor_a", "family_id": "family_a", "name": "Anchor A"}],
                        "disconnections": [
                            {
                                "id": "duplicate_move",
                                "family_id": "family_a",
                                "name": "Move A",
                                "retrosynthetic_move": {"planner_hint": "hint"},
                                "evidence": [{"type": "literature", "citation": "missing url"}],
                                "use_policy": {"proposal_source": "unit"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "schema_version": "unit.b",
                        "families": [{"family_id": "family_b", "name": "Family B"}],
                        "anchors": [{"anchor_id": "anchor_b", "family_id": "family_b", "name": "Anchor B"}],
                        "disconnections": [
                            {
                                "id": "duplicate_move",
                                "family_id": "family_b",
                                "name": "Move B",
                                "retrosynthetic_move": {"planner_hint": "hint"},
                                "evidence": [{"type": "literature", "citation": "ok", "url": "https://example.test"}],
                                "use_policy": {"proposal_source": "unit"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_databases([first, second])

        self.assertFalse(report["passed"])
        self.assertGreaterEqual(report["issue_count"], 2)
        self.assertEqual(report["duplicate_ids"]["disconnections"][0]["id"], "duplicate_move")
        self.assertEqual(len(report["evidence"]["evidence_items_without_url"]), 1)
        self.assertEqual(len(report["evidence"]["evidence_items_without_trace"]), 1)


def _load_source_records():
    records = {
        "families": [],
        "anchors": [],
        "disconnections": [],
    }
    for path in sorted(Path().glob(DB_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        for section in records:
            records[section].extend((path, item) for item in data.get(section, []) or [])
    return records


if __name__ == "__main__":
    unittest.main()
