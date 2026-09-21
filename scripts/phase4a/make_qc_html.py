#!/usr/bin/env python3
"""Build a browser QC page: image + statements + checklist for 300 pairs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QC_DIR = ROOT / "data/phase4a/qc"
TSV = QC_DIR / "review_300.tsv"
IMG_DIR = QC_DIR / "images"
HTML = QC_DIR / "review.html"

CHECKS = [
    ("pos_true", "1. 正例为真", "图中这条 Positive 关系是否成立？"),
    ("neg_natural", "2. 负例句法自然", "Negative 这句话读起来是否像正常人写的英文？"),
    ("neg_plausible", "3. 负例语义合理", "不看图的话，这句话在某个现实场景里有没有可能发生？"),
    ("neg_visually_false", "4. 负例在图中不成立", "看图后，Negative 关系是否确实不成立？"),
    ("needs_image", "5. 必须看图才能判 No", "如果只看文字就能凭常识判断为假，填否。"),
    ("is_synonym", "6. 是正例同义/近义", "Negative 是否其实和 Positive 一个意思？"),
    ("role_reversal", "7. 只是主客体对调", "是否只是把 subject/object 反过来了？"),
    ("ambiguous", "8. 图中无法判断", "看不清、角度歧义、无法确定？"),
    ("overall_clean", "9. 总体可作干净样本", "建议：1–5 为是，6–8 为否，才标是。"),
]


def main() -> None:
    with TSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    missing = 0
    for i, rec in enumerate(rows, start=1):
        src = Path(rec["image_path"])
        name = src.name
        dest = IMG_DIR / name
        if src.exists():
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            os.symlink(src, dest)
        else:
            missing += 1
        payload.append(
            {
                "n": i,
                "id": rec["id"],
                "family": rec.get("family") or "",
                "image_file": name,
                "image_path": rec["image_path"],
                "subject": rec.get("subject") or "",
                "object": rec.get("object") or "",
                "positive_relation": rec.get("positive_relation") or "",
                "negative_relation": rec.get("negative_relation") or "",
                "positive_statement": rec.get("positive_statement") or "",
                "negative_statement": rec.get("negative_statement") or "",
                "old_negative_statement": rec.get("old_negative_statement") or "",
                "candidate_rank": rec.get("candidate_rank") or "",
            }
        )

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__CHECKS__", json.dumps(CHECKS, ensure_ascii=False))
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} n={len(payload)} missing_images={missing}", flush=True)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 4A 人工 QC（300）</title>
<style>
  :root { --bg:#0f1419; --card:#1a2330; --ink:#e8eef6; --muted:#9aa8b7; --yes:#2f9e73; --no:#c45c5c; --acc:#4c8dff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:#121a24; border-bottom:1px solid #2a3646; padding:12px 20px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .progress { color:var(--muted); font-size:13px; }
  button { background:#243044; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:8px 12px; cursor:pointer; }
  button:hover { background:#2d3d52; }
  button.primary { background:var(--acc); border-color:var(--acc); color:#fff; }
  main { max-width:1180px; margin:0 auto; padding:20px; }
  .card { background:var(--card); border-radius:14px; padding:16px; display:grid; grid-template-columns: minmax(280px, 1fr) 1.15fr; gap:18px; }
  @media (max-width: 900px) { .card { grid-template-columns: 1fr; } }
  img { width:100%; max-height:520px; object-fit:contain; background:#0b0f14; border-radius:10px; }
  .meta { color:var(--muted); font-size:12px; margin:8px 0 0; word-break:break-all; }
  .stmt { padding:10px 12px; border-radius:10px; margin:0 0 10px; }
  .stmt b { display:block; font-size:12px; margin-bottom:4px; letter-spacing:.02em; }
  .pos { background:#173528; }
  .neg { background:#3a1f24; }
  .old { background:#243044; color:var(--muted); }
  .sro { font-size:13px; color:var(--muted); margin-bottom:12px; }
  .q { border-top:1px solid #2a3646; padding-top:8px; margin-top:8px; }
  .q .title { font-size:14px; }
  .q .hint { font-size:12px; color:var(--muted); margin:2px 0 6px; }
  .yn { display:flex; gap:8px; }
  .yn label { display:flex; align-items:center; gap:6px; padding:6px 10px; border-radius:8px; border:1px solid #3a4a5e; cursor:pointer; }
  .yn label.on-yes { border-color:var(--yes); background:#173528; }
  .yn label.on-no { border-color:var(--no); background:#3a1f24; }
  textarea { width:100%; min-height:54px; background:#121a24; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:8px; }
  .jump { width:64px; background:#121a24; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:7px 8px; }
  .nav { display:flex; gap:8px; align-items:center; }
  .imgroot { flex:1; min-width:280px; background:#121a24; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:7px 10px; }
  .hintbar { width:100%; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Phase 4A 人工 QC</h1>
  <div class="progress" id="progress"></div>
  <div class="nav">
    <button id="prev">上一张</button>
    <input class="jump" id="jump" type="number" min="1" max="300">
    <button id="next">下一张</button>
  </div>
  <button class="primary" id="save">导出填写结果 TSV</button>
  <button id="clear">清空本机填写</button>
  <input class="imgroot" id="imgroot" spellcheck="false" title="本地 RelSim 图片目录">
  <div class="hintbar">本地打开：图片默认 D:\works\relation_hallu\data\relsim_images。若 Chrome 不显示图，用 Edge/Firefox，或把本 html 拷进该图片目录并把路径改成 .</div>
</header>
<main>
  <div class="card" id="card"></div>
</main>
<script>
const ITEMS = __DATA__;
const CHECKS = __CHECKS__;
const KEY = "phase4a_qc_v1";
const ROOT_KEY = "phase4a_qc_imgroot";
const DEFAULT_ROOT = "D:/works/relation_hallu/data/relsim_images";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = 0;

function imageSrc(name) {
  let root = (document.getElementById("imgroot").value || DEFAULT_ROOT).trim();
  root = root.replace(/\\/g, "/");
  if (!root || root === ".") return name;
  if (!root.endsWith("/")) root += "/";
  if (/^[A-Za-z]:\//.test(root)) return "file:///" + root + name;
  return root + name;
}

function saveStore() {
  localStorage.setItem(KEY, JSON.stringify(store));
}
function recId() { return ITEMS[idx].id; }
function val(field) { return (store[recId()] || {})[field] || ""; }
function setVal(field, v) {
  store[recId()] = store[recId()] || {};
  store[recId()][field] = v;
  saveStore();
  render();
}
function filledCount() {
  return ITEMS.filter(it => (store[it.id] || {}).overall_clean).length;
}
function render() {
  const it = ITEMS[idx];
  document.getElementById("jump").value = idx + 1;
  document.getElementById("progress").textContent =
    `${idx+1} / ${ITEMS.length}　family=${it.family}　已填 overall ${filledCount()}/300`;
  const ans = store[it.id] || {};
  const checks = CHECKS.map(([k, title, hint]) => {
    const v = ans[k] || "";
    return `<div class="q">
      <div class="title">${title}</div>
      <div class="hint">${hint}</div>
      <div class="yn">
        <label class="${v==="1"?"on-yes":""}"><input type="radio" name="${k}" value="1" ${v==="1"?"checked":""}> 是</label>
        <label class="${v==="0"?"on-no":""}"><input type="radio" name="${k}" value="0" ${v==="0"?"checked":""}> 否</label>
      </div>
    </div>`;
  }).join("");
  document.getElementById("card").innerHTML = `
    <div>
      <img src="${imageSrc(it.image_file)}" alt="${it.id}" onerror="this.nextElementSibling.style.display='block'">
      <p class="meta" style="display:none;color:#c45c5c">图片未加载：${imageSrc(it.image_file)}<br>请检查上方本地目录，或把本 html 拷到图片文件夹并把目录改成 .</p>
      <p class="meta">${it.id}<br>${it.image_file}</p>
    </div>
    <div>
      <div class="sro">subject = <b style="color:#e8eef6">${it.subject}</b>
        &nbsp;→&nbsp; object = <b style="color:#e8eef6">${it.object}</b></div>
      <div class="stmt pos"><b>正例（应成立）　${it.positive_relation}</b>${it.positive_statement}</div>
      <div class="stmt neg"><b>负例（应不成立，且语言上要像样）　${it.negative_relation}</b>${it.negative_statement}</div>
      <div class="stmt old"><b>旧 dirty 负例（只对照，不用评）</b>${it.old_negative_statement}</div>
      ${checks}
      <div class="q"><div class="title">备注</div>
        <textarea id="notes">${ans.notes || ""}</textarea>
      </div>
    </div>`;
  document.querySelectorAll(".yn input").forEach(el => {
    el.addEventListener("change", () => setVal(el.name, el.value));
  });
  document.getElementById("notes").addEventListener("input", (e) => {
    store[recId()] = store[recId()] || {};
    store[recId()].notes = e.target.value;
    saveStore();
  });
}
function exportTsv() {
  const fields = ["id","image_path","family","subject","positive_relation","object","positive_statement","negative_relation","negative_statement","old_negative_statement","candidate_rank","pos_true","neg_natural","neg_plausible","neg_visually_false","needs_image","is_synonym","role_reversal","ambiguous","overall_clean","notes"];
  const lines = [fields.join("\t")];
  for (const it of ITEMS) {
    const a = store[it.id] || {};
    const row = {
      ...it,
      pos_true: a.pos_true || "",
      neg_natural: a.neg_natural || "",
      neg_plausible: a.neg_plausible || "",
      neg_visually_false: a.neg_visually_false || "",
      needs_image: a.needs_image || "",
      is_synonym: a.is_synonym || "",
      role_reversal: a.role_reversal || "",
      ambiguous: a.ambiguous || "",
      overall_clean: a.overall_clean || "",
      notes: (a.notes || "").replace(/\t|\n/g, " "),
    };
    lines.push(fields.map(k => row[k] || "").join("\t"));
  }
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/tab-separated-values"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_300_filled.tsv";
  a.click();
  URL.revokeObjectURL(url);
}
document.getElementById("prev").onclick = () => { idx = (idx + ITEMS.length - 1) % ITEMS.length; render(); };
document.getElementById("next").onclick = () => { idx = (idx + 1) % ITEMS.length; render(); };
document.getElementById("jump").onchange = (e) => {
  const n = Math.min(ITEMS.length, Math.max(1, parseInt(e.target.value || "1", 10)));
  idx = n - 1; render();
};
document.getElementById("imgroot").value = localStorage.getItem(ROOT_KEY) || DEFAULT_ROOT;
document.getElementById("imgroot").addEventListener("change", () => {
  localStorage.setItem(ROOT_KEY, document.getElementById("imgroot").value);
  render();
});
document.getElementById("save").onclick = exportTsv;
document.getElementById("clear").onclick = () => {
  if (confirm("清空本机保存的填写？")) { localStorage.removeItem(KEY); location.reload(); }
};
document.addEventListener("keydown", (e) => {
  if (["INPUT","TEXTAREA"].includes(e.target.tagName)) return;
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
  if (e.key === "ArrowRight") document.getElementById("next").click();
});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
