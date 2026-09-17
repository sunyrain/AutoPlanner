# 当前模型输出协议与工具定义原文

整理日期：2026-09-16。主册：[BASE_PROMPTS_CURRENT.md](BASE_PROMPTS_CURRENT.md)。这些是模型可见输出 schema，不是 Host 事后添加的 artifact 包装。字段、required、additionalProperties、枚举、长度和数组限制均由当前代码直接生成；不做手工简化。

JSON schema 通过独立输出参数约束模型，不需要再把它当成普通 prompt 全文复制一次。中文标题不是模型指令。这里的“修复”及“物料边界”版本由 host_context 开关控制。

源码：[cascade_planner/agent/codex_worker.py:2614](../../cascade_planner/agent/codex_worker.py#L2614)

## 策略组合生成

任务：`paper_matched_strategy_generator`；artifact：`StrategyPortfolioReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_cards": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "strategy_query": {
            "type": "string"
          },
          "critical_assumption": {
            "type": "string"
          },
          "critic_checkpoint": {
            "type": "string"
          }
        },
        "required": [
          "strategy_query",
          "critical_assumption",
          "critic_checkpoint"
        ]
      },
      "minItems": 0
    }
  },
  "required": [
    "strategy_cards"
  ]
}
```

## 冻结版三策略组合生成

任务：`paper_matched_strategy_generator`；artifact：`StrategyPortfolioReport`；host 开关：`{"strategy_count": 3}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_cards": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "strategy_query": {
            "type": "string"
          },
          "critical_assumption": {
            "type": "string"
          },
          "critic_checkpoint": {
            "type": "string"
          }
        },
        "required": [
          "strategy_query",
          "critical_assumption",
          "critic_checkpoint"
        ]
      },
      "minItems": 3,
      "maxItems": 3
    }
  },
  "required": [
    "strategy_cards"
  ]
}
```

## 策略组合审查

任务：`paper_matched_strategy_critic`；artifact：`StrategyPortfolioReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_cards": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "strategy_query": {
            "type": "string"
          },
          "critical_assumption": {
            "type": "string"
          },
          "critic_checkpoint": {
            "type": "string"
          },
          "review_decision": {
            "type": "string",
            "enum": [
              "keep",
              "revise",
              "replace",
              "discard"
            ]
          },
          "decisive_risk": {
            "type": "string",
            "minLength": 1
          }
        },
        "required": [
          "strategy_query",
          "critical_assumption",
          "critic_checkpoint",
          "review_decision",
          "decisive_risk"
        ]
      },
      "minItems": 0
    }
  },
  "required": [
    "strategy_cards"
  ]
}
```

## 上游单策略生成

任务：`paper_matched_strategy_generator`；artifact：`StrategyCardReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_query": {
      "type": "string"
    },
    "critical_assumption": {
      "type": "string"
    },
    "critic_checkpoint": {
      "type": "string"
    }
  },
  "required": [
    "strategy_query",
    "critical_assumption",
    "critic_checkpoint"
  ]
}
```

## 带物料边界选择的上游生成

任务：`paper_matched_strategy_generator`；artifact：`StrategyCardReport`；host 开关：`{"allow_material_boundary": true}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_query": {
      "type": "string"
    },
    "critical_assumption": {
      "type": "string"
    },
    "critic_checkpoint": {
      "type": "string"
    },
    "material_boundary": {
      "anyOf": [
        {
          "type": "null"
        },
        {
          "type": "object",
          "properties": {
            "material_name": {
              "type": "string",
              "minLength": 1,
              "maxLength": 180
            },
            "material_kind": {
              "type": "string",
              "enum": [
                "defined_compound",
                "polymer",
                "mixture",
                "biological_material"
              ]
            },
            "reference_ids": {
              "type": "array",
              "minItems": 1,
              "maxItems": 4,
              "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 180
              }
            },
            "rationale": {
              "type": "string",
              "minLength": 1,
              "maxLength": 420
            },
            "unresolved_requirements": {
              "type": "array",
              "minItems": 1,
              "maxItems": 4,
              "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240
              }
            }
          },
          "required": [
            "material_name",
            "material_kind",
            "reference_ids",
            "rationale",
            "unresolved_requirements"
          ],
          "additionalProperties": false
        }
      ]
    }
  },
  "required": [
    "strategy_query",
    "critical_assumption",
    "critic_checkpoint",
    "material_boundary"
  ]
}
```

## 上游单策略审查

任务：`paper_matched_strategy_critic`；artifact：`StrategyCardReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "strategy_query": {
      "type": "string"
    },
    "critical_assumption": {
      "type": "string"
    },
    "critic_checkpoint": {
      "type": "string"
    },
    "review_decision": {
      "type": "string",
      "enum": [
        "keep",
        "revise",
        "replace"
      ]
    },
    "decisive_risk": {
      "type": "string",
      "minLength": 1
    }
  },
  "required": [
    "strategy_query",
    "critical_assumption",
    "critic_checkpoint",
    "review_decision",
    "decisive_risk"
  ]
}
```

## 一步 Builder

任务：`paper_matched_route_step`；artifact：`RetrosynthesisProposalReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "checkpoint_relation": {
      "type": "string",
      "enum": [
        "preparatory",
        "executes_checkpoint"
      ]
    },
    "reaction_intent": {
      "type": "string",
      "maxLength": 300
    },
    "execution_domain": {
      "type": "string",
      "enum": [
        "chemical",
        "enzymatic",
        "whole_cell",
        "hybrid"
      ]
    },
    "catalyst": {
      "type": "string"
    },
    "continuation_hint": {
      "type": "string"
    },
    "reaction_operations": {
      "type": "array",
      "items": {
        "anyOf": [
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "break_bond"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "add_bond"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_bond_order"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "delta": {
                "type": "number"
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "delta"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_atom"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "formal_charge": {
                "type": "integer"
              }
            },
            "required": [
              "op",
              "map_idx",
              "formal_charge"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_atom"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "isotope": {
                "type": "integer",
                "minimum": 0
              }
            },
            "required": [
              "op",
              "map_idx",
              "isotope"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_explicit_h"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "count": {
                "type": "integer",
                "minimum": 0
              },
              "no_implicit": {
                "type": "boolean"
              }
            },
            "required": [
              "op",
              "map_idx",
              "count",
              "no_implicit"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "add_group"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "fragment_smiles": {
                "type": "string"
              }
            },
            "required": [
              "op",
              "map_idx",
              "fragment_smiles"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "remove_group"
                ]
              },
              "map_indices": {
                "type": "array",
                "items": {
                  "type": "integer",
                  "minimum": 1
                },
                "minItems": 1
              }
            },
            "required": [
              "op",
              "map_indices"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "invert_stereocenter"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_idx"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "clear_stereocenter"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_idx"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_bond_stereo"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "stereo": {
                "type": "string",
                "enum": [
                  "NONE",
                  "ANY"
                ]
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "stereo"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_bond_stereo"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "stereo": {
                "type": "string",
                "enum": [
                  "Z",
                  "E",
                  "CIS",
                  "TRANS"
                ]
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "stereo"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_tetrahedral_stereo"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "configuration": {
                "type": "string",
                "enum": [
                  "R",
                  "S"
                ]
              }
            },
            "required": [
              "op",
              "map_idx",
              "configuration"
            ]
          }
        ]
      },
      "minItems": 1
    },
    "conditions": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 4
    }
  },
  "required": [
    "checkpoint_relation",
    "reaction_intent",
    "execution_domain",
    "catalyst",
    "continuation_hint",
    "reaction_operations",
    "conditions"
  ]
}
```

## 修复 Builder

任务：`paper_matched_route_step`；artifact：`RetrosynthesisProposalReport`；host 开关：`{"allow_repair_recovery": true}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "recovery": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "expand",
            "backtrack",
            "expand_scope"
          ]
        },
        "step_id": {
          "type": "string",
          "maxLength": 160
        },
        "reason": {
          "type": "string",
          "maxLength": 500
        }
      },
      "required": [
        "action",
        "step_id",
        "reason"
      ]
    },
    "checkpoint_relation": {
      "type": "string",
      "enum": [
        "preparatory",
        "executes_checkpoint"
      ]
    },
    "reaction_intent": {
      "type": "string",
      "maxLength": 300
    },
    "execution_domain": {
      "type": "string",
      "enum": [
        "chemical",
        "enzymatic",
        "whole_cell",
        "hybrid"
      ]
    },
    "catalyst": {
      "type": "string"
    },
    "continuation_hint": {
      "type": "string"
    },
    "reaction_operations": {
      "type": "array",
      "items": {
        "anyOf": [
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "break_bond"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "add_bond"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_bond_order"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "delta": {
                "type": "number"
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "delta"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_atom"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "formal_charge": {
                "type": "integer"
              }
            },
            "required": [
              "op",
              "map_idx",
              "formal_charge"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "change_atom"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "isotope": {
                "type": "integer",
                "minimum": 0
              }
            },
            "required": [
              "op",
              "map_idx",
              "isotope"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_explicit_h"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "count": {
                "type": "integer",
                "minimum": 0
              },
              "no_implicit": {
                "type": "boolean"
              }
            },
            "required": [
              "op",
              "map_idx",
              "count",
              "no_implicit"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "add_group"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "fragment_smiles": {
                "type": "string"
              }
            },
            "required": [
              "op",
              "map_idx",
              "fragment_smiles"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "remove_group"
                ]
              },
              "map_indices": {
                "type": "array",
                "items": {
                  "type": "integer",
                  "minimum": 1
                },
                "minItems": 1
              }
            },
            "required": [
              "op",
              "map_indices"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "invert_stereocenter"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_idx"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "clear_stereocenter"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              }
            },
            "required": [
              "op",
              "map_idx"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_bond_stereo"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "stereo": {
                "type": "string",
                "enum": [
                  "NONE",
                  "ANY"
                ]
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "stereo"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_bond_stereo"
                ]
              },
              "map_a": {
                "type": "integer",
                "minimum": 1
              },
              "map_b": {
                "type": "integer",
                "minimum": 1
              },
              "stereo": {
                "type": "string",
                "enum": [
                  "Z",
                  "E",
                  "CIS",
                  "TRANS"
                ]
              }
            },
            "required": [
              "op",
              "map_a",
              "map_b",
              "stereo"
            ]
          },
          {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "op": {
                "type": "string",
                "enum": [
                  "set_tetrahedral_stereo"
                ]
              },
              "map_idx": {
                "type": "integer",
                "minimum": 1
              },
              "configuration": {
                "type": "string",
                "enum": [
                  "R",
                  "S"
                ]
              }
            },
            "required": [
              "op",
              "map_idx",
              "configuration"
            ]
          }
        ]
      },
      "minItems": 0
    },
    "conditions": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 4
    }
  },
  "required": [
    "recovery",
    "checkpoint_relation",
    "reaction_intent",
    "execution_domain",
    "catalyst",
    "continuation_hint",
    "reaction_operations",
    "conditions"
  ]
}
```

## 关键事件 Critic

任务：`paper_matched_key_event_critic`；artifact：`ChemicalStrategyCritique`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "checkpoint_match": {
      "type": "boolean"
    },
    "verdict": {
      "type": "string",
      "enum": [
        "pass",
        "uncertain",
        "reject"
      ]
    },
    "uncertainty_source": {
      "type": [
        "string",
        "null"
      ],
      "enum": [
        null,
        "proposal_underspecified",
        "evidence_missing",
        "assessment_unresolved"
      ]
    },
    "blocking_type": {
      "type": "string",
      "enum": [
        "none",
        "structure",
        "missing_reactive_handle",
        "mechanism",
        "atom_provenance",
        "conditions",
        "functional_group_compatibility",
        "chemoselectivity",
        "stereochemistry",
        "sequence_dependency",
        "competing_pathway"
      ]
    },
    "repair_scope": {
      "type": "string",
      "enum": [
        "none",
        "focus_edge",
        "route_span",
        "strategy_horizon"
      ]
    },
    "required_change_kind": {
      "type": "string",
      "enum": [
        "none",
        "conditions_or_catalyst",
        "precursor_covalent_state",
        "reaction_topology",
        "strategy_horizon"
      ]
    },
    "competing_site_maps": {
      "type": "array",
      "items": {
        "type": "integer",
        "minimum": 1
      },
      "maxItems": 8
    },
    "reasons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 2
    },
    "suggested_revision": {
      "type": "string"
    }
  },
  "required": [
    "checkpoint_match",
    "verdict",
    "uncertainty_source",
    "blocking_type",
    "repair_scope",
    "required_change_kind",
    "competing_site_maps",
    "reasons",
    "suggested_revision"
  ]
}
```

## 整路线 Critic

任务：`paper_matched_route_critic`；artifact：`ChemicalStrategyCritique`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "route_overall_evaluation": {
      "type": "string"
    },
    "strategy_adherence": {
      "type": "boolean"
    },
    "step_assessments": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "review_slot": {
            "type": "string",
            "maxLength": 32
          },
          "verdict": {
            "type": "string",
            "enum": [
              "pass",
              "uncertain",
              "reject"
            ]
          },
          "uncertainty_source": {
            "type": [
              "string",
              "null"
            ],
            "enum": [
              null,
              "proposal_underspecified",
              "evidence_missing",
              "assessment_unresolved"
            ]
          },
          "blocking_type": {
            "type": "string",
            "enum": [
              "none",
              "structure",
              "missing_reactive_handle",
              "mechanism",
              "atom_provenance",
              "conditions",
              "functional_group_compatibility",
              "chemoselectivity",
              "stereochemistry",
              "sequence_dependency",
              "competing_pathway"
            ]
          },
          "reasons": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "maxItems": 2
          },
          "condition_assessment": {
            "type": "string"
          },
          "suggested_revision": {
            "type": "string"
          }
        },
        "required": [
          "review_slot",
          "verdict",
          "uncertainty_source",
          "blocking_type",
          "reasons",
          "condition_assessment",
          "suggested_revision"
        ]
      },
      "minItems": 1,
      "maxItems": 32
    },
    "route_level_risks": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 4
    },
    "repair_actions": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 4
    },
    "coupled_blocker_groups": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "string",
          "maxLength": 160
        },
        "minItems": 2
      }
    },
    "chemical_dependencies": {
      "type": "array",
      "maxItems": 6,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "consumer_review_slot": {
            "type": "string",
            "maxLength": 32
          },
          "prerequisite_review_slots": {
            "type": "array",
            "items": {
              "type": "string",
              "maxLength": 32
            },
            "maxItems": 8
          },
          "requirement": {
            "type": "string",
            "maxLength": 320
          }
        },
        "required": [
          "consumer_review_slot",
          "prerequisite_review_slots",
          "requirement"
        ]
      }
    },
    "limitations": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 2
    }
  },
  "required": [
    "route_overall_evaluation",
    "strategy_adherence",
    "step_assessments",
    "route_level_risks",
    "repair_actions",
    "coupled_blocker_groups",
    "chemical_dependencies",
    "limitations"
  ]
}
```

## 当前 Path Editor

任务：`path_repair_editor`；artifact：`RetrosynthesisProposalReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "change_step_ids": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 160
      },
      "maxItems": 25
    },
    "additional_coupled_blocker_step_ids": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 160
      }
    },
    "preserved_suffix_compatible": {
      "type": "boolean"
    },
    "repair_goal": {
      "type": "string",
      "maxLength": 500
    },
    "active_constraints": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 240
      },
      "maxItems": 5
    }
  },
  "required": [
    "change_step_ids",
    "additional_coupled_blocker_step_ids",
    "preserved_suffix_compatible",
    "repair_goal",
    "active_constraints"
  ]
}
```

## 兼容 RouteJSON Editor

任务：`paper_matched_route_editor`；artifact：`RetrosynthesisProposalReport`；host 开关：`{}`。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "repair_summary": {
      "type": "string",
      "maxLength": 500
    },
    "replace_span": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "remove_step_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "maxLength": 160
          },
          "minItems": 1,
          "maxItems": 25
        },
        "revised_steps": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "step_id": {
                "type": "string",
                "maxLength": 160
              },
              "product_smiles": {
                "type": "string"
              },
              "reaction_family": {
                "type": "string",
                "maxLength": 160
              },
              "conditions": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "maxItems": 4
              },
              "execution_domain": {
                "type": "string",
                "enum": [
                  "chemical",
                  "enzymatic",
                  "whole_cell",
                  "hybrid"
                ]
              },
              "catalyst": {
                "type": "string"
              },
              "reaction_operations": {
                "type": "array",
                "items": {
                  "anyOf": [
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "break_bond"
                          ]
                        },
                        "map_a": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "map_b": {
                          "type": "integer",
                          "minimum": 1
                        }
                      },
                      "required": [
                        "op",
                        "map_a",
                        "map_b"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "add_bond"
                          ]
                        },
                        "map_a": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "map_b": {
                          "type": "integer",
                          "minimum": 1
                        }
                      },
                      "required": [
                        "op",
                        "map_a",
                        "map_b"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "change_bond_order"
                          ]
                        },
                        "map_a": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "map_b": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "delta": {
                          "type": "number"
                        }
                      },
                      "required": [
                        "op",
                        "map_a",
                        "map_b",
                        "delta"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "change_atom"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "formal_charge": {
                          "type": "integer"
                        }
                      },
                      "required": [
                        "op",
                        "map_idx",
                        "formal_charge"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "change_atom"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "isotope": {
                          "type": "integer",
                          "minimum": 0
                        }
                      },
                      "required": [
                        "op",
                        "map_idx",
                        "isotope"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "set_explicit_h"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "count": {
                          "type": "integer",
                          "minimum": 0
                        },
                        "no_implicit": {
                          "type": "boolean"
                        }
                      },
                      "required": [
                        "op",
                        "map_idx",
                        "count",
                        "no_implicit"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "add_group"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "fragment_smiles": {
                          "type": "string"
                        }
                      },
                      "required": [
                        "op",
                        "map_idx",
                        "fragment_smiles"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "remove_group"
                          ]
                        },
                        "map_indices": {
                          "type": "array",
                          "items": {
                            "type": "integer",
                            "minimum": 1
                          },
                          "minItems": 1
                        }
                      },
                      "required": [
                        "op",
                        "map_indices"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "invert_stereocenter"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        }
                      },
                      "required": [
                        "op",
                        "map_idx"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "clear_stereocenter"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        }
                      },
                      "required": [
                        "op",
                        "map_idx"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "set_bond_stereo"
                          ]
                        },
                        "map_a": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "map_b": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "stereo": {
                          "type": "string",
                          "enum": [
                            "NONE",
                            "ANY"
                          ]
                        }
                      },
                      "required": [
                        "op",
                        "map_a",
                        "map_b",
                        "stereo"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "set_bond_stereo"
                          ]
                        },
                        "map_a": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "map_b": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "stereo": {
                          "type": "string",
                          "enum": [
                            "Z",
                            "E",
                            "CIS",
                            "TRANS"
                          ]
                        }
                      },
                      "required": [
                        "op",
                        "map_a",
                        "map_b",
                        "stereo"
                      ]
                    },
                    {
                      "type": "object",
                      "additionalProperties": false,
                      "properties": {
                        "op": {
                          "type": "string",
                          "enum": [
                            "set_tetrahedral_stereo"
                          ]
                        },
                        "map_idx": {
                          "type": "integer",
                          "minimum": 1
                        },
                        "configuration": {
                          "type": "string",
                          "enum": [
                            "R",
                            "S"
                          ]
                        }
                      },
                      "required": [
                        "op",
                        "map_idx",
                        "configuration"
                      ]
                    }
                  ]
                },
                "minItems": 1
              }
            },
            "required": [
              "step_id",
              "product_smiles",
              "reaction_family",
              "conditions",
              "execution_domain",
              "catalyst",
              "reaction_operations"
            ]
          },
          "minItems": 1,
          "maxItems": 25
        }
      },
      "required": [
        "remove_step_ids",
        "revised_steps"
      ]
    }
  },
  "required": [
    "repair_summary",
    "replace_span"
  ]
}
```

<a id="tools"></a>

## 工具定义原文

定义来自 `application/chemistry_inspection_mcp.py`。query 的操作枚举和参数随当前可用操作裁剪；本段仅计算定义，不启动 MCP 服务，不调用网络。

### inspect_mapped_smiles

```json
{
  "name": "inspect_mapped_smiles",
  "description": "Inspect mapped SMILES with local RDKit and return mapped atom facts, local adjacency and bond orders, ring paths, CIP centers, stereo bonds, and optionally a bounded enumeration of unassigned stereoisomers.",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "smiles": {
        "type": "string"
      },
      "map_ids": {
        "type": "array",
        "items": {
          "type": "integer",
          "minimum": 1
        },
        "maxItems": 16
      },
      "enumerate_unassigned": {
        "type": "boolean"
      },
      "max_isomers": {
        "type": "integer",
        "minimum": 1,
        "maximum": 32
      }
    },
    "required": [
      "smiles"
    ]
  }
}
```

### query_planning_evidence：stock, list

```json
{
  "name": "query_planning_evidence",
  "description": "Read-only, Host-budgeted queries. Available operations: stock, list. Use list to reuse discoveries. No reaction proof or route mutation.",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "operation": {
        "type": "string",
        "enum": [
          "stock",
          "list"
        ]
      },
      "smiles": {
        "type": "string",
        "maxLength": 6000
      }
    },
    "required": [
      "operation"
    ]
  }
}
```

### query_planning_evidence：stock, compound, search, read, list

```json
{
  "name": "query_planning_evidence",
  "description": "Read-only, Host-budgeted queries. Available operations: stock, compound, search, read, list. Use list to reuse discoveries. No reaction proof or route mutation.",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "operation": {
        "type": "string",
        "enum": [
          "stock",
          "compound",
          "search",
          "read",
          "list"
        ]
      },
      "query": {
        "type": "string",
        "maxLength": 800
      },
      "smiles": {
        "type": "string",
        "maxLength": 6000
      },
      "source_id": {
        "type": "string",
        "maxLength": 800
      }
    },
    "required": [
      "operation"
    ]
  }
}
```
