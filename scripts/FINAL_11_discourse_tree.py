import json
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "article03"

ANSWER_FIT_FILE = OUTPUT_DIR / "answer_fit_scores.json"
RELATION_FILE = OUTPUT_DIR / "final_qud_relations_refined.json"

if not RELATION_FILE.exists():
    RELATION_FILE = OUTPUT_DIR / "final_qud_relations.json"

EDU_FILE = OUTPUT_DIR / "edus.json"

TREE_JSON = OUTPUT_DIR / "discourse_tree.json"
TREE_TXT = OUTPUT_DIR / "discourse_tree.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_records(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "records", "relations", "predictions",
            "results", "data", "candidates"
        ]:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def get_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_relation(record):
    for key in [
        "final_relation", "refined_relation",
        "predicted_relation", "relation",
        "relation_label", "predicted_label"
    ]:
        if record.get(key) is not None:
            return str(record[key])

    return "Unknown"


def get_confidence(record):
    for key in [
        "final_confidence", "refined_confidence",
        "relation_confidence", "confidence",
        "prediction_confidence"
    ]:
        if record.get(key) is not None:
            return get_number(record[key])

    return 0.0


def find_edu_texts(data):
    """
    Supports EDU lists containing strings or dictionaries.
    EDU IDs are treated as 1-based positions.
    """
    if isinstance(data, dict):
        for key in ["edus", "records", "sentences", "data"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Unsupported edus.json structure.")

    texts = {}

    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            texts[index] = item

        elif isinstance(item, dict):
            edu_id = item.get("edu_id", item.get("id", index))
            text = item.get("text", item.get("edu_text", ""))

            if text:
                texts[int(edu_id)] = text

    return texts


# ============================================================
# UNION-FIND: PREVENT CYCLES
# ============================================================

class UnionFind:

    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])

        return self.parent[node]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return True


# ============================================================
# MAIN TREE CONSTRUCTION
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 11: BOTTOM-UP DISCOURSE TREE CONSTRUCTION")
    print("=" * 65)

    print("\nLoading EDU texts...")
    edu_texts = find_edu_texts(load_json(EDU_FILE))

    print("Total EDUs:", len(edu_texts))

    print("\nLoading answer-fit scores...")
    answer_data = load_json(ANSWER_FIT_FILE)
    answer_records = get_records(answer_data)

    print("Answer-fit records:", len(answer_records))

    print("\nLoading relation predictions...")
    relation_data = load_json(RELATION_FILE)
    relation_records = get_records(relation_data)

    print("Relation records:", len(relation_records))

    # --------------------------------------------------------
    # 1. Index relation predictions by EDU pair
    # --------------------------------------------------------

    relation_lookup = {}

    for record in relation_records:
        edu1 = record.get("edu1")
        edu2 = record.get("edu2")

        if edu1 is None or edu2 is None:
            continue

        try:
            key = (int(edu1), int(edu2))
        except (TypeError, ValueError):
            continue

        relation_lookup[key] = record

    # --------------------------------------------------------
    # 2. Select the best QUD candidate for each EDU pair
    # --------------------------------------------------------

    pair_candidates = defaultdict(list)

    for record in answer_records:

        edu1 = record.get("edu1")
        edu2 = record.get("edu2")

        if edu1 is None or edu2 is None:
            continue

        try:
            edu1 = int(edu1)
            edu2 = int(edu2)
        except (TypeError, ValueError):
            continue

        if edu1 not in edu_texts or edu2 not in edu_texts:
            continue

        if edu1 == edu2:
            continue

        answer_fit = get_number(record.get("answer_fit_score"))
        salience = get_number(record.get("salience_score"))

        # Select candidates using salience x answer-fit.
        selection_score = salience * max(0.0, min(1.0, answer_fit))

        pair_candidates[(edu1, edu2)].append({
            "edu1": edu1,
            "edu2": edu2,
            "qud": record.get("qud", ""),
            "salience": salience,
            "answer_fit": answer_fit,
            "selection_score": selection_score,
            "pair_id": record.get("pair_id"),
            "candidate_index": record.get("candidate_index")
        })

    # --------------------------------------------------------
    # 3. Create weighted merge edges
    # --------------------------------------------------------

    print("\nSelecting QUD candidates and scoring merge edges...")

    edges = []

    for (edu1, edu2), candidates in pair_candidates.items():

        # Highest salience x answer-fit candidate
        selected = max(
            candidates,
            key=lambda item: item["selection_score"]
        )

        relation_record = relation_lookup.get((edu1, edu2), {})

        # Try the reverse direction if the forward direction is absent.
        if not relation_record:
            relation_record = relation_lookup.get((edu2, edu1), {})

        relation = get_relation(relation_record)
        confidence = get_confidence(relation_record)

        # Clamp confidence to [0, 1].
        confidence = max(0.0, min(1.0, confidence))

        # Combined merge score.
        merge_score = (
            selected["salience"]
            * max(0.0, min(1.0, selected["answer_fit"]))
            * confidence
        )

        edges.append({
            "edu1": edu1,
            "edu2": edu2,
            "pair_id": selected["pair_id"],
            "qud": selected["qud"],
            "relation": relation,
            "salience": selected["salience"],
            "answer_fit": selected["answer_fit"],
            "relation_confidence": confidence,
            "merge_score": merge_score
        })

    # Highest-confidence edges are considered first.
    edges.sort(
        key=lambda edge: edge["merge_score"],
        reverse=True
    )

    print("Unique EDU-pair edges:", len(edges))

    # --------------------------------------------------------
    # 4. Greedy bottom-up merging
    # --------------------------------------------------------

    print("\nBuilding discourse tree...")

    nodes = set(edu_texts.keys())
    union_find = UnionFind(nodes)

    # Initially, every EDU is a leaf.
    components = {
        edu_id: {
            "node_id": f"EDU_{edu_id}",
            "type": "EDU",
            "edu_id": edu_id,
            "text": text
        }
        for edu_id, text in edu_texts.items()
    }

    # Each current component is represented by a root EDU.
    component_roots = {edu_id: edu_id for edu_id in nodes}

    tree_nodes = []
    accepted_edges = []
    rejected_edges = []

    merge_count = 0

    for edge in edges:

        edu1 = edge["edu1"]
        edu2 = edge["edu2"]

        if not union_find.union(edu1, edu2):
            rejected_edges.append({
                **edge,
                "reason": "Would create a cycle"
            })
            continue

        root1 = component_roots[union_find.find(edu1)]
        root2 = component_roots[union_find.find(edu2)]

        # Recover the actual component nodes.
        left_node = components[root1]
        right_node = components[root2]

        merge_count += 1

        merged_node = {
            "node_id": f"MERGE_{merge_count}",
            "type": "DISCOURSE_RELATION",
            "relation": edge["relation"],
            "qud": edge["qud"],
            "merge_score": round(edge["merge_score"], 6),
            "salience_score": round(edge["salience"], 6),
            "answer_fit_score": round(edge["answer_fit"], 6),
            "relation_confidence": round(
                edge["relation_confidence"], 6
            ),
            "left": left_node,
            "right": right_node
        }

        tree_nodes.append(merged_node)

        accepted_edges.append({
            **edge,
            "merge_node": merged_node["node_id"]
        })

        # Track components using Union-Find's updated root.
        new_root = union_find.find(edu1)

        components.pop(root1, None)
        components.pop(root2, None)

        components[new_root] = merged_node
        component_roots[new_root] = new_root

        # Remove obsolete component-root mappings.
        for key in list(component_roots.keys()):
            if union_find.find(key) == new_root:
                component_roots[key] = new_root

    # --------------------------------------------------------
    # 5. Identify final root(s)
    # --------------------------------------------------------

    final_roots = []

    for node_id in nodes:
        root = union_find.find(node_id)

        if root in components and root not in final_roots:
            final_roots.append(root)

    final_trees = [components[root] for root in final_roots]

    complete_tree = len(final_trees) == 1

    result = {
        "stage": 11,
        "stage_name": "Bottom-Up Discourse Tree Construction",
        "number_of_edus": len(edu_texts),
        "candidate_edges": len(edges),
        "accepted_merges": len(accepted_edges),
        "rejected_cycle_edges": len(rejected_edges),
        "number_of_final_components": len(final_trees),
        "complete_tree": complete_tree,
        "scoring_formula": (
            "salience_score x answer_fit_score x relation_confidence"
        ),
        "selection_method": (
            "Highest salience x answer-fit candidate per EDU pair"
        ),
        "trees": final_trees,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges
    }

    # --------------------------------------------------------
    # 6. Save JSON
    # --------------------------------------------------------

    with open(TREE_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 7. Save readable text report
    # --------------------------------------------------------

    def write_tree(node, file, depth=0):

        indent = "  " * depth

        if node["type"] == "EDU":
            file.write(
                f"{indent}EDU {node['edu_id']}: {node['text']}\n"
            )
            return

        file.write(
            f"{indent}{node['node_id']} | "
            f"Relation: {node['relation']} | "
            f"Score: {node['merge_score']:.4f}\n"
        )

        file.write(f"{indent}QUD: {node['qud']}\n")

        write_tree(node["left"], file, depth + 1)
        write_tree(node["right"], file, depth + 1)

    with open(TREE_TXT, "w", encoding="utf-8") as file:

        file.write("STAGE 11: BOTTOM-UP DISCOURSE TREE\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Total EDUs: {len(edu_texts)}\n")
        file.write(f"Candidate edges: {len(edges)}\n")
        file.write(f"Accepted merges: {len(accepted_edges)}\n")
        file.write(f"Rejected cycle edges: {len(rejected_edges)}\n")
        file.write(f"Final components: {len(final_trees)}\n")
        file.write(f"Complete tree: {complete_tree}\n\n")

        for index, tree in enumerate(final_trees, start=1):
            file.write(f"\nFINAL TREE / COMPONENT {index}\n")
            file.write("-" * 65 + "\n")
            write_tree(tree, file)

        file.write("\n\nACCEPTED MERGES\n")
        file.write("=" * 65 + "\n")

        for edge in accepted_edges:
            file.write(
                f"EDU {edge['edu1']} + EDU {edge['edu2']} | "
                f"{edge['relation']} | "
                f"Score: {edge['merge_score']:.4f}\n"
            )

        file.write("\nREJECTED EDGES\n")
        file.write("=" * 65 + "\n")

        for edge in rejected_edges:
            file.write(
                f"EDU {edge['edu1']} + EDU {edge['edu2']} | "
                f"{edge['reason']}\n"
            )

    # --------------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STAGE 11 COMPLETED")
    print("=" * 65)

    print("EDUs:", len(edu_texts))
    print("Candidate edges:", len(edges))
    print("Accepted merges:", len(accepted_edges))
    print("Rejected cycle edges:", len(rejected_edges))
    print("Final components:", len(final_trees))
    print("Complete tree:", complete_tree)

    print("\nJSON saved:", TREE_JSON)
    print("TXT saved:", TREE_TXT)

    if not complete_tree:
        print(
            "\nWARNING: A single complete tree could not be constructed "
            "from the available edges. Inspect the final components "
            "and rejected edges."
        )


if __name__ == "__main__":
    main()