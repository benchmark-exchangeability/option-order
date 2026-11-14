import csv
import json
import math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).parent
data = root / "data"
nominal = 1 / 24


def rows(name):
    with (data / name).open() as f:
        return list(csv.DictReader(f))


def beta_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d, h = 1, 1 - qab * x / qap, 1
    d = 1 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
        step = d * c
        h *= step
        if abs(step - 1) < 3e-14:
            break
    return h


def beta_cdf(x, a, b):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    return factor * beta_fraction(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - factor * beta_fraction(b, a, 1 - x) / b


def beta_quantile(p, a, b):
    lo, hi = 0, 1
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if beta_cdf(mid, a, b) < p else (lo, mid)
    return (lo + hi) / 2


def exact_interval(k, n):
    return [0 if k == 0 else beta_quantile(.025, k, n - k + 1), 1 if k == n else beta_quantile(.975, k + 1, n - k)]


def binomial_probability(k, n, p):
    if p == 0:
        return float(k == 0)
    if p == 1:
        return float(k == n)
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_test(k, n, p=.5, alternative="two-sided"):
    probabilities = [binomial_probability(x, n, p) for x in range(n + 1)]
    if alternative == "greater":
        return min(1, sum(probabilities[k:]))
    observed = probabilities[k]
    return min(1, sum(value for value in probabilities if value <= observed * (1 + 1e-12)))


def fisher_greater(a, b, c, d):
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    low, high = max(0, row1 - (total - col1)), min(row1, col1)
    denominator = math.comb(total, row1)
    return sum(math.comb(col1, x) * math.comb(total - col1, row1 - x) / denominator for x in range(max(a, low), high + 1))


def fisher_power(n, k, alpha=.05, target=.8):
    log_comb = [math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1) for x in range(n + 1)]
    critical = {}
    for total in range(2 * n + 1):
        low, high = max(0, total - n), min(n, total)
        logs = [log_comb[y] + log_comb[total - y] for y in range(low, high + 1)]
        peak = max(logs)
        weights = [math.exp(value - peak) for value in logs]
        denominator = sum(weights)
        tail = 0
        for index in range(len(weights) - 1, -1, -1):
            tail += weights[index]
            if tail / denominator <= alpha:
                critical[total] = low + index
            else:
                break
    reference = [binomial_probability(x, n, k / n) for x in range(n + 1)]
    rejection = [sum(reference[x] for x in range(n + 1) if y >= critical.get(x + y, n + 1)) for y in range(n + 1)]

    def probability(delta):
        return sum(binomial_probability(y, n, k / n + delta) * rejection[y] for y in range(n + 1))

    low, high = 0, .1
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if probability(middle) < target else (low, middle)
    return {"n": n, "reference_fires": k, "reference_rate": k / n, "alpha": alpha, "null_rejection_probability": probability(0), "target_power": target, "delta": high, "delta_points": 100 * high}


def rate(values):
    values = list(map(int, values))
    n, k = len(values), sum(values)
    return {"n": n, "fires": k, "rate": k / n, "ci95": exact_interval(k, n), "fold": k / n / nominal}


def mcnemar(a, b, alternative="two-sided"):
    first_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    second_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n = first_only + second_only
    p = binomial_test(second_only, n, .5, alternative) if n else 1
    return {"first_arm_only": first_only, "second_arm_only": second_only, "p": p}


pooled = rows("pooled_items.csv")
for row in pooled:
    for key in ["fire", "num_monotone", "anchor_last", "boolean_set", "alpha", "charlen"]:
        row[key] = int(row[key])


def item_class(row):
    for key in ["boolean_set", "num_monotone", "anchor_last"]:
        if row[key]:
            return key
    return "key_monotone" if row["alpha"] or row["charlen"] else "residual"


def pass_screen(row, keys=()):
    blocked = row["boolean_set"] or row["num_monotone"] or row["anchor_last"]
    blocked |= any(row[key] for key in keys)
    return not blocked


corpus_order = ["MMLU (27 subjects)", "MedMCQA", "ARC-Challenge", "professional_medicine", "clinical_knowledge", "MedQA", "OpenBookQA"]
table1 = {name: rate(row["fire"] for row in pooled if row["corpus"] == name) for name in corpus_order}

pooled_by_uid = {row["item_uid"]: row for row in pooled}
model_rows = rows("model_items.csv")
models = defaultdict(dict)
for row in model_rows:
    models[row["model"]][row["item_uid"]] = int(row["fire"])
model_order = ["gpt2", "gpt2-large", "gpt2-xl", "Qwen2.5-1.5B", "Qwen2.5-3B", "BioGPT-Large", "BioMedLM"]
shared = sorted(models["gpt2-large"])
retained = [uid for uid in shared if pass_screen(pooled_by_uid[uid], ("alpha", "charlen"))]
reference_all = [models["gpt2-large"][uid] for uid in shared]
reference_retained = [models["gpt2-large"][uid] for uid in retained]
table2 = {}
for model in model_order:
    published = [models[model][uid] for uid in shared]
    screened = [models[model][uid] for uid in retained]
    result = {"published": rate(published), "p_nominal": binomial_test(sum(published), len(published), nominal, "greater"), "screened": rate(screened)}
    if model != "gpt2-large":
        k, n, r = sum(screened), len(screened), sum(reference_retained)
        result["gap_points"] = 100 * (k - r) / n
        result["p_fisher"] = fisher_greater(k, n - k, r, n - r)
        result["mcnemar"] = mcnemar(reference_retained, screened, "greater")
    table2[model] = result

intervention = rows("intervention_items.csv")
by_arm = defaultdict(dict)
for row in intervention:
    by_arm[row["arm"]][row["source_id"]] = int(row["fire"])
arm_order = ["A-0", "A-PARA", "A-SHORT", "A-ANCH", "A-ANCHFIRST", "A-NUM"]
table3 = {arm: rate(by_arm[arm].values()) for arm in arm_order}
comparisons = [("A-PARA", "A-0"), ("A-ANCH", "A-SHORT"), ("A-ANCH", "A-ANCHFIRST"), ("A-NUM", "A-0")]
table3_pairs = {}
for treatment, control in comparisons:
    ids = sorted(set(by_arm[treatment]) & set(by_arm[control]))
    before = [by_arm[control][uid] for uid in ids]
    after = [by_arm[treatment][uid] for uid in ids]
    table3_pairs[f"{treatment} vs {control}"] = {"difference_points": 100 * (sum(after) - sum(before)) / len(ids), **mcnemar(after, before)}

classes = defaultdict(list)
for row in pooled:
    classes[item_class(row)].append(row["fire"])
total_excess = sum(int(row["fire"]) - nominal for row in pooled)
table4 = {}
for name in ["boolean_set", "num_monotone", "anchor_last", "key_monotone", "residual"]:
    result = rate(classes[name])
    result["prevalence"] = result["n"] / len(pooled)
    result["excess_share"] = sum(x - nominal for x in classes[name]) / total_excess
    table4[name] = result

power = fisher_power(1445, 112)
table5 = sorted([{key: value if key == "subject" else int(value) if key in ["n", "ascending"] else float(value) for key, value in row.items()} for row in rows("census_mmlu.csv")], key=lambda row: row["fold"], reverse=True)
table6 = [{key: value if key in ["benchmark", "source"] else int(value) if key in ["k", "n", "ascending"] else float(value) for key, value in row.items()} for row in rows("census_benchmarks.csv")]

reference_rates = {name: {key: value for key, value in result.items() if key != "fires"} for name, result in table1.items()}
models_output = {}
for name, result in table2.items():
    output = {"published_rate": result["published"]["rate"], "p_nominal": result["p_nominal"], "screened_rate": result["screened"]["rate"]}
    if name != "gpt2-large":
        output |= {"gap_points": result["gap_points"], "p_fisher": result["p_fisher"], "p_mcnemar": result["mcnemar"]["p"]}
    models_output[name] = output
classes_output = {name: {key: value for key, value in result.items() if key not in ["fires", "ci95"]} for name, result in table4.items()}
results = {"reference_rates": reference_rates, "models": models_output, "intervention": {"arms": table3, "pairs": table3_pairs}, "classes": classes_output, "power": power, "mmlu_census": table5, "benchmark_census": table6}

assert table1["MMLU (27 subjects)"]["fires"] == 855
assert table2["gpt2-large"]["published"]["fires"] == 252
assert table2["Qwen2.5-3B"]["screened"]["fires"] == 110
assert table3["A-NUM"]["fires"] == 898
assert table4["residual"]["n"] == 11027
assert round(table4["residual"]["excess_share"], 3) == .382
assert round(power["null_rejection_probability"], 4) == .0433
assert round(power["delta"], 4) == .0273

(root / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps({"outputs": list(results), "file": str(root / "results.json")}, indent=2))
