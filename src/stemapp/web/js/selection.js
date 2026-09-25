// stem の ON/OFF（選択）の規則。DOM や音声には触れない純粋な処理だけを置く。
//
// 親子のルール:
// - 選択の状態は「葉」（そのジョブで子を持たない stem）の集合だけで持つ。
// - 子に分かれている親（例: vocals → lead_vocal / backing_vocal）の音は鳴らさない（GainNode は常に 0）。
//   子の合計＝親なので、子を全部鳴らせば親と同じ音になる。
// - 親のボタンは「子をまとめて ON/OFF」する。子が全部 ON なら親は ON、一部なら「一部」表示。
//   押すと、全部 ON のときは全部 OFF、それ以外は全部 ON にする。
// - グループ・組み合わせプリセットに親が含まれていれば、その子（葉）すべてを指すとみなす。

/** stems（API の /api/jobs/{id}/stems の stems）から木を作る。 */
export function buildTree(stems) {
  const byCode = new Map();
  const children = new Map();
  for (const s of stems) {
    byCode.set(s.code, s);
    children.set(s.code, []);
  }
  for (const s of stems) {
    if (s.parent_code && children.has(s.parent_code)) children.get(s.parent_code).push(s.code);
  }
  const order = stems.map((s) => s.code);
  const leaves = order.filter((c) => children.get(c).length === 0);
  return { byCode, children, order, leaves };
}

/** code の葉（子を持たなければ自分）。このジョブに無い code なら空。 */
export function leavesOf(tree, code) {
  if (!tree.byCode.has(code)) return [];
  const kids = tree.children.get(code);
  if (!kids.length) return [code];
  return kids.flatMap((k) => leavesOf(tree, k));
}

export function isParent(tree, code) {
  return (tree.children.get(code) || []).length > 0;
}

/** 葉の集合 codes が選択 sel の中でどうなっているか: "on" / "off" / "partial"。 */
export function stateOfLeaves(sel, leaves) {
  if (!leaves.length) return "off";
  let on = 0;
  for (const c of leaves) if (sel.has(c)) on++;
  if (on === 0) return "off";
  return on === leaves.length ? "on" : "partial";
}

export function stateOf(tree, sel, code) {
  return stateOfLeaves(sel, leavesOf(tree, code));
}

/** 葉の集合をまとめて切り替える（全部 ON なら全部 OFF、それ以外は全部 ON）。新しい Set を返す。 */
export function toggleLeaves(sel, leaves) {
  const next = new Set(sel);
  const allOn = leaves.length > 0 && leaves.every((c) => sel.has(c));
  for (const c of leaves) {
    if (allOn) next.delete(c);
    else next.add(c);
  }
  return next;
}

export function toggle(tree, sel, code) {
  return toggleLeaves(sel, leavesOf(tree, code));
}

/** ソロ: その stem（親なら子すべて）だけを ON にする。 */
export function solo(tree, code) {
  return new Set(leavesOf(tree, code));
}

/** 全部: 元の曲と同じ（葉をすべて ON）。 */
export function allOn(tree) {
  return new Set(tree.leaves);
}

/** グループ（members は stem の code の配列）の葉。このジョブに無いメンバーは無視する。 */
export function groupLeaves(tree, group) {
  const out = [];
  for (const code of group.members || []) {
    for (const leaf of leavesOf(tree, code)) if (!out.includes(leaf)) out.push(leaf);
  }
  return out;
}

export function dbToGain(db) {
  return Math.pow(10, (Number(db) || 0) / 20);
}

/**
 * 組み合わせプリセットを選択にする。groups は /api/stem-groups の stem_groups。
 * 戻り値: { sel: Set（葉）, gainsDb: Map（葉 → dB。複数の項目に入る葉は後の項目が優先） }
 */
export function presetToSelection(tree, preset, groups) {
  const byGroupCode = new Map(groups.map((g) => [g.code, g]));
  const byGroupId = new Map(groups.map((g) => [g.group_id, g]));
  const sel = new Set();
  const gainsDb = new Map();
  for (const item of preset.items || []) {
    let leaves = [];
    if (item.stem_type_code) {
      leaves = leavesOf(tree, item.stem_type_code);
    } else {
      const g = byGroupCode.get(item.group_code) || byGroupId.get(item.group_id);
      if (g) leaves = groupLeaves(tree, g);
    }
    for (const leaf of leaves) {
      sel.add(leaf);
      gainsDb.set(leaf, Number(item.gain_db) || 0);
    }
  }
  return { sel, gainsDb };
}

/**
 * 今の選択を組み合わせプリセットの items にする（stem_type_id と gain_db）。
 * 子が全部 ON の親は、親1つにまとめる（親が詳細分割されていない曲でも同じ意味になるように）。
 * stemTypeIdOf: code → stem_type_id。
 */
export function selectionToItems(tree, sel, stemTypeIdOf, gainsDb = new Map()) {
  const items = [];
  const walk = (code) => {
    const state = stateOf(tree, sel, code);
    if (state === "off") return;
    const kids = tree.children.get(code);
    const uniformGain = leavesOf(tree, code).every(
      (c) => (gainsDb.get(c) || 0) === (gainsDb.get(leavesOf(tree, code)[0]) || 0),
    );
    if (state === "on" && (kids.length === 0 || uniformGain)) {
      const id = stemTypeIdOf(code);
      if (id !== undefined && id !== null) {
        items.push({ stem_type_id: id, gain_db: gainsDb.get(leavesOf(tree, code)[0]) || 0 });
        return;
      }
    }
    for (const k of kids) walk(k);
  };
  const roots = tree.order.filter((c) => {
    const s = tree.byCode.get(c);
    return !s.parent_code || !tree.byCode.has(s.parent_code);
  });
  for (const r of roots) walk(r);
  return items;
}

/** 各 stem の音量（倍率）。子に分かれた親は 0、葉は選択中なら dB の倍率、そうでなければ 0。 */
export function targetGains(tree, sel, gainsDb = new Map()) {
  const out = {};
  for (const code of tree.order) {
    if (isParent(tree, code)) out[code] = 0;
    else out[code] = sel.has(code) ? dbToGain(gainsDb.get(code) || 0) : 0;
  }
  return out;
}

/** 選択が同じか（葉の集合として）。 */
export function sameSelection(a, b) {
  if (a.size !== b.size) return false;
  for (const c of a) if (!b.has(c)) return false;
  return true;
}
