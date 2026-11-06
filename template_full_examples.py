########################GENIA###########################

ENTITY_TYPE_genia = {
    "protein": "Proteins are large, complex molecules composed of amino acids that perform various functions in organisms, such as structural support, catalysis, and signaling.",
    "rna": "RNA (ribonucleic acid) is a nucleic acid involved in coding, decoding, regulation, and expression of genes, often acting as a messenger between DNA and proteins.",
    "cell type": "A cell type refers to a distinct form of cell characterized by specific structure and function, such as neurons, monocytes, or T cells.",
    "cell line": "A cell line is a population of cells derived from a single cell and grown in vitro, which can proliferate indefinitely under specific conditions.",
    "dna": "DNA (deoxyribonucleic acid) is the hereditary material in almost all living organisms, encoding genetic instructions for development and function."}

PROMPT_TEMPLATE_genia = """You are an excellent linguist. Given the following entity types, please complete the entity extraction task and output the results as required:
Entities List:
1. protein: Proteins are large, complex molecules composed of amino acids that perform various functions in organisms, such as structural support, catalysis, and signaling.
2. rna: RNA (ribonucleic acid) is a nucleic acid involved in coding, decoding, regulation, and expression of genes, often acting as a messenger between DNA and proteins.
3. cell type: A cell type refers to a distinct form of cell characterized by specific structure and function, such as neurons, monocytes, or T cells.
4. cell line: A cell line is a population of cells derived from a single cell and grown in vitro, which can proliferate indefinitely under specific conditions.
5. dna: DNA (deoxyribonucleic acid) is the hereditary material in almost all living organisms, encoding genetic instructions for development and function.

Notes:
Output entity extraction results according to the following format:
```json[{"type": "xx", "entity": "x"}, {"type": "xx", "entity": "xx"}, ...]```
- If there are no valid entities, output an empty list ```json[]```."""

#######################Fusion##############################

PROMPT_example_1 = """You are an expert in few-shot prompt optimization. Your task is to incrementally refine entity extraction instructions by selecting and updating high-quality annotation examples.

---
### Input Data

Current Entity Definitions:
{}

Candidate Examples (newly generated or collected examples):
{}

---

### Task Requirements

1. Verify the Candidate Examples based strictly on the Current Entity Definitions. An example is valid ONLY IF annotated labels strictly match the entity definitions and unambiguous contextual evidence exists.

2. Refine the Example Set INCREMENTALLY and SAFELY:
   - ANCHOR STABILITY (CRITICAL): Keep fundamental, standard examples intact. Do NOT drastically replace the entire example set. 
   - DIVERSITY GAIN: Only add or substitute an existing example if the new candidate introduces a distinct, high-value edge case (e.g., implicit expressions, previously confused boundaries) without duplicating existing structural triggers.
   - NO REDUNDANCY: Keep only the single most representative example for a specific contextual scenario.
   - Maintain a highly curated set of examples. The `Context` text MUST BE UNIQUE. 

3. Output ONLY the unified instruction in the specified format below. DO NOT include analysis steps, intermediate verification steps, or explanations.

### Critical Output Rules

- MUST include all entity labels in the definitions section: {}
- The output MUST consist of exactly two sections: "## Entity Definitions" and "## Annotation Examples".
- Do NOT put examples inside the Entity Definitions section. List all verified examples together at the end.

### Output Format

## Entity Definitions
1. [Entity Name]: [Definition]
2. [Entity Name]: [Definition]
...

## Annotation Examples
- 1. "[Context Text]" -> Answer: [{{"type": "xx", "entity": "x"}}, {{"type": "xx", "entity": "x"}}, ...]
..."""

PROMPT_compare_1 = """You are an expert in entity definition refinement. Your task is to incrementally optimize entity definitions based on confusion cases and provide illustrative annotation examples.

---
### Input Data

Current Entity Definitions:
{}

Confusion Examples (complete entity annotations with error information):
{}

---

### Task Requirements

1. Analyze the confusion examples. Each example shows the COMPLETE entity list for a text, along with error information indicating which entities were confused.

2. Refine the definitions INCREMENTALLY and SAFELY:
   - CONCISENESS: Keep definitions extremely concise. Do not create bloated definitions with endless appended clauses.
   - ANTI-OVERFITTING: Do not overfit to the specific confusion examples. The revised definition must remain universally applicable to standard cases.
   - ELEGANT MERGING: If adding distinguishing features, merge them elegantly into the core definition. Only use "Distinguish from:" if absolutely necessary.
   - NO NEW CONFLICTS: Ensure that your adjustments to separate two entities do not accidentally overlap with the definition of a THIRD entity.

3. Select and update Annotation Examples based on the Confusion Examples:
   - Select examples that best illustrate the corrected boundaries.
   - STABILITY: Preserve existing basic/classic example contexts that illustrate standard cases. However, if a preserved context also appears in the Confusion Examples, its entity spans and labels MUST be updated according to the complete true annotations in the Confusion Examples. Only replace an example context if the new one provides a significantly higher diversity gain for distinguishing boundaries.
   - All newly added or updated examples must use the exact Context and Span from the Confusion Examples. Entities for each context must be complete. For each span, keep only one of the most correct span label, and it must come from the true labels in the given examples.
   - Maintain a highly curated set of examples. The `Context` text MUST BE UNIQUE. 

4. Output ONLY the refined definitions and the selected examples in the specified format below. DO NOT include analysis steps or explanations.

### Critical Output Rules

- MUST include all entity labels in the definitions section: {}
- If "Confusion Examples" is empty or "None", return the current definitions and existing examples unchanged.
- Output format MUST consist of exactly two sections: "## Entity Definitions" and "## Entity Examples".

### Output Format

## Entity Definitions
1. [Entity Name]: [Core Definition integrated with distinguishing boundaries if necessary]. [Optional: Distinguish from [Confusing Entity]: [Brief feature].]
2. [Entity Name]: [Core Definition integrated with distinguishing boundaries if necessary]. [Optional: Distinguish from [Confusing Entity]: [Brief feature].]
...

## Entity Examples
- 1. "[Context Text]" -> Span: "[Target Span]" is identified as [Entity Name]; Span: "[Target Span]" is identified as [Entity Name]; ...
..."""

PROMPT_parse_1 = """You are an expert in entity semantic decomposition. Your task is to incrementally expand entity definitions by identifying fine-grained sub-semantic categories from recall error cases.

---
### Input Data

Current Entity Definitions:
{}

Recall Error Examples (complete entity annotations with error information):
{}

---

### Task Requirements

1. Analyze the recall error examples. Each example shows the COMPLETE entity list for a text, along with error information indicating which entities were missed.

2. Refine the definitions INCREMENTALLY and SAFELY:
   - CONFLICT AVOIDANCE (CRITICAL): Before adding a sub-semantic category, verify that it does not logically overlap with the definitions of OTHER existing entities. Do not introduce ambiguity.
   - CONCISENESS: Merge sub-semantics elegantly. Avoid long, verbose lists.
   - ANTI-OVERFITTING: Ensure the abstracted dimensions represent a true class of entities, not just a quirk of the provided examples.

3. Select and update Annotation Examples based on the Recall Error Examples:
   - Select examples that best illustrate the newly added sub-semantic categories.
   - STABILITY: Preserve core example contexts when useful; if a preserved context appears in Recall Error Examples, update its spans/labels to the complete true annotations there. Replace a context only to better cover new sub-semantics or clear diversity gain.
   - All newly added or updated examples must use the exact Context and Span from the Recall Error Examples. Entities for each context must be complete. For each span, keep only one of the most correct span label, and it must come from the true labels in the given examples.
   - Maintain a highly curated set of examples. The `Context` text MUST BE UNIQUE. 


4. Output ONLY the refined definitions and the selected examples in the specified format below. DO NOT include analysis steps or explanations.

### Critical Output Rules

- MUST include all entity labels in the definitions section: {}
- If "Recall Error Examples" is empty or "None", return the current definitions and existing examples unchanged.
- Output format MUST consist of exactly two sections: "## Entity Definitions" and "## Entity Examples".

### Output Format

## Entity Definitions
1. [Entity Name]: [Core Definition]. [If sub-semantics found: Includes generalized aspects like [Category 1] and [Category 2].]
2. [Entity Name]: [Core Definition]. [If sub-semantics found: Includes generalized aspects like [Category 1] and [Category 2].]
...

## Entity Examples
- 1. "[Context Text]" -> Span: "[Target Span]" is identified as [Entity Name]; Span: "[Target Span]" is identified as [Entity Name]; ...
..."""


# ---------------------------------------------------------------------------
# 差异化冷启动模板：仅依据现有定义改写，不引入新语义（第 0 轮 Compare/Parsing 专家使用）
# 两个模板均只有两个 {} 占位：1) 当前定义文本  2) 必须保留的实体标签列表
# ---------------------------------------------------------------------------
PROMPT_compare_init = """You are an expert in entity definition refinement. Your task is to rewrite a set of entity definitions so that the boundaries between easily confused entity types become explicit, using ONLY the semantics already present in the definitions.

---
### Input Data

Current Entity Definitions:
{}

---

### Task Requirements

1. NO NEW SEMANTICS (CRITICAL): Every statement in your output must be derivable from the current definitions. Do NOT add new facts, new sub-categories, new example terms, or outside knowledge. You may only rephrase, reorder, merge, and make explicit what the definitions already state or directly imply (for example an exclusion or priority rule such as "label it as X instead of Y").
2. Find pairs of types whose current definitions are adjacent or overlapping (for example one is a more specific form of another, or both could apply to the same span). For each such pair, state the discriminating feature using only wording taken from those two definitions, in the form "Distinguish from [Other Entity]: [feature]".
3. Types with no confusable neighbor MUST stay unchanged.
4. Keep every example term already given in a definition (such as "e.g." or "such as" items). Do not add or remove any.
5. CONCISENESS: Keep each definition short. Do not append long clauses.
6. Do NOT add annotation examples or any extra section.
7. Write in the same language as the current definitions.

### Critical Output Rules

- MUST include all entity labels, spelled exactly as in the current definitions: {}
- Output ONLY the section below. DO NOT include analysis or explanations.

### Output Format

## Entity Definitions
1. [Entity Name]: [Definition]. [Optional: Distinguish from [Other Entity]: [feature].]
2. [Entity Name]: [Definition]. [Optional: Distinguish from [Other Entity]: [feature].]
..."""

PROMPT_parse_init = """You are an expert in entity semantic decomposition. Your task is to rewrite a set of entity definitions so that the sub-semantic aspects each definition already covers are made explicit, using ONLY the semantics already present in the definitions.

---
### Input Data

Current Entity Definitions:
{}

---

### Task Requirements

1. NO NEW SEMANTICS (CRITICAL): Every statement in your output must be derivable from the current definitions. Do NOT add new facts, new sub-categories, new example terms, or outside knowledge. You may only split, rephrase, and enumerate what a definition already states or directly implies.
2. For each definition, decompose its existing wording into sub-semantic aspects, such as the kinds, forms, or surface expressions it already lists or implies (for example items given after "e.g." or "such as", or phrases joined by "and", "or", "including"). Rewrite it as: core definition + "Includes aspects like [aspect 1], [aspect 2]", with every aspect taken from the original wording.
3. If a definition has no separable sub-semantics, keep it unchanged.
4. Keep every example term already given in a definition. Do not add or remove any.
5. CONFLICT AVOIDANCE: Do not let an enumerated aspect overlap with the definition of another entity. If the original wording already routes it to another type, keep that routing.
6. CONCISENESS: Keep each definition short. Do not append long clauses.
7. Do NOT add annotation examples or any extra section.
8. Write in the same language as the current definitions.

### Critical Output Rules

- MUST include all entity labels, spelled exactly as in the current definitions: {}
- Output ONLY the section below. DO NOT include analysis or explanations.

### Output Format

## Entity Definitions
1. [Entity Name]: [Core Definition]. [If separable: Includes aspects like [aspect 1], [aspect 2].]
2. [Entity Name]: [Core Definition]. [If separable: Includes aspects like [aspect 1], [aspect 2].]
..."""

Fusion_system_test = """Your goal is to complete information extraction tasks based on multi-expert perspectives.

## The Three Experts:

1.  Expert A (Comparative Analysis): Specializes in distinguishing between easily confused entity types. Prioritize Expert A’s type assignment when two experts disagree on the label for the same span.
    *   Definitions: {}

2.  Expert B (Semantic Decomposition): Specializes in recalling entities that are easily missed due to broad or implicit semantics. Prioritize Expert B’s recall when deciding whether a singleton prediction is a genuine entity.
    *   Definitions: {}

3.  Expert C (Exemplar Inference): Specializes in context-based identification using annotated examples. Use Expert C as a tiebreaker when A and B conflict, or to verify ambiguous spans.
    *   Definitions: {}
"""

Fusion_user_test = """---
## Target Text:
{}
---

## Final Output Format:

Output only the JSON list. Do not include explanation.
```json[{{"type": "xx", "entity": "x"}}, ...]```
- If there are no valid entities, output: ```json[]```."""

Fusion_system_train = """Your goal is to complete information extraction tasks based on multi-expert definitions and their predictions.

## Experts Definitions And Their Predictions:

1.  Expert A (Comparative Analysis): Specializes in distinguishing between easily confused entity types. Prioritize Expert A’s type assignment when two experts disagree on the label for the same span.
    *   Definitions: {}
    *   Predictions: {}

2.  Expert B (Semantic Decomposition): Specializes in recalling entities that are easily missed due to broad or implicit semantics. Prioritize Expert B’s recall when deciding whether a singleton prediction is a genuine entity.
    *   Definitions: {}
    *   Predictions: {}

3.  Expert C (Exemplar Inference): Specializes in context-based identification using annotated examples. Use Expert C as a tiebreaker when A and B conflict, or to verify ambiguous spans.
    *   Definitions: {}
    *   Predictions: {}
"""

Fusion_user_train = """---
## Target Text:
{}
---

## Final Output Format:

Output only the JSON list. Do not include explanation.
```json[{{"type": "xx", "entity": "x"}}, ...]```
- If there are no valid entities, output: ```json[]```."""

Expert_prompt_template = """You are an excellent linguist. Given the following entity types, please complete the entity extraction task and output the results as required:
Entities List:
{}

Notes:
Output entity extraction results according to the following format:
```json[{{"type": "xx", "entity": "x"}}, {{"type": "xx", "entity": "xx"}}, ...]```
- If there are no valid entities, output an empty list ```json[]```."""