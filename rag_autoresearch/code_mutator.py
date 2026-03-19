import hashlib
from pathlib import Path


def code_signature(file_paths):
    hasher = hashlib.sha256()
    for path in file_paths:
        data = Path(path).read_bytes()
        hasher.update(data)
        hasher.update(b"|")
    return hasher.hexdigest()


def apply_mutation(mutation):
    originals = {}
    for edit in mutation.get("edits", []):
        path = Path(edit["file"])
        content = path.read_text(encoding="utf-8")
        search = edit["search"]
        replace = edit["replace"]

        if search not in content:
            return {"ok": False, "error": f"Search string not found in {path}"}
        if content.count(search) > 1:
            return {"ok": False, "error": f"Search string not unique in {path}"}

        originals[str(path)] = content
        path.write_text(content.replace(search, replace), encoding="utf-8")

    for edit in mutation.get("edits", []):
        path = Path(edit["file"])
        try:
            compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            rollback(originals)
            return {"ok": False, "error": f"Syntax error in {path}: {exc}"}

    return {"ok": True, "originals": originals}


def rollback(originals):
    for path, content in originals.items():
        Path(path).write_text(content, encoding="utf-8")


def pick_mutation(config, rng, mutation_stats=None, temperature=0.3):
    pool = set(config.get("code_mutation_pool", []))
    mutations = config.get("code_mutations", [])
    if not mutations:
        return None

    if pool:
        candidates = [m for m in mutations if m.get("id") in pool]
    else:
        candidates = mutations

    if not candidates:
        return None

    if not mutation_stats:
        return rng.choice(candidates)

    weights = []
    for mutation in candidates:
        entry = mutation_stats.get(mutation.get("id"), {})
        avg_delta = entry.get("avg_delta", 0.0)
        weights.append(pow(2.718281828, avg_delta / max(temperature, 1e-6)))

    total = sum(weights)
    cutoff = rng.random() * total if total > 0 else None
    running = 0.0
    for mutation, weight in zip(candidates, weights):
        running += weight
        if cutoff is None or running >= cutoff:
            return mutation
    return candidates[-1]
