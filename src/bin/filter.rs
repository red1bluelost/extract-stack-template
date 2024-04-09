use arrow::array::{self, array::StringArray, FixedSizeListBuilder, RecordBatch, StringBuilder, ArrayRef};
use futures::TryStreamExt;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use regex::{Regex, RegexBuilder};
use tokio::{fs, sync::mpsc};

use std::{collections::HashMap, sync::Arc};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

const NUM_INSTANTS: usize = 5;

type CandidateVec<'s> = arrayvec::ArrayVec<&'s str, NUM_INSTANTS>;

lazy_static::lazy_static! {
    static ref TEMPLATE_RE: Regex = Regex::new(r"^template *<[^>]+>").unwrap();
    static ref INSTANTS_RE: Regex =
        RegexBuilder::new(r"^Instantiation \d+:\n```(cpp|c\+\+)?\n(.*?)\n```(.*)$")
        .dot_matches_new_line(true)
        .build()
        .unwrap();
}

fn get_param_count(func_def: tree_sitter::Node) -> Option<usize> {
    debug_assert!(func_def.kind() == "function_definiton");

    let decl = func_def.child_by_field_name("declarator")?;
    let params = decl.child_by_field_name("parameters")?;
    Some(params.named_child_count())
}

async fn filter_instantiations(
    tx: mpsc::Sender<(String, [String; 5])>,
    record: RecordBatch,
) -> anyhow::Result<()> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_cpp::language())
        .map_err(extract_template::into_parquet_error)?;

    let [elements, templates] = record.columns() else {
        panic!("should have two elements")
    };
    let templates = array::as_string_array(templates);
    let elements = array::as_list_array(elements);
    let templates = templates.into_iter().flatten();
    let elements = elements.iter().flatten();
    for (template, element_list) in templates.zip(elements) {
        let tpl_tree = parser
            .parse(template, None)
            .expect("template should not fail parse");

        let tpl_root = tpl_tree.root_node();
        if tpl_root.child_count() != 1 {
            continue;
        }

        let tpl_child = tpl_root.child(0).expect("should have one child");
        if tpl_child.kind() != "template_declaration" {
            continue;
        }

        let mut tpl_cursor = tpl_child.walk();
        let Some(tpl_func_def) = tpl_child
            .named_children(&mut tpl_cursor)
            .find(|n| n.kind() == "function_definition")
        else {
            continue;
        };

        let Some(tpl_param_count) = get_param_count(tpl_func_def) else {
            continue;
        };

        let elm_list = array::as_string_array(&element_list);
        let es: Vec<_> = elm_list.into_iter().flatten().collect();
        let elms: [_; 4] = es.try_into().expect("should have four elements");

        let mut map = HashMap::new();
        for e in elms {
            let mut cursor = e.trim();
            while let Some(cap) = INSTANTS_RE.captures(cursor) {
                let code = cap
                    .get(2)
                    .expect("should have second capture")
                    .as_str()
                    .trim();
                cursor = cap
                    .get(3)
                    .expect("should have third capture")
                    .as_str()
                    .trim();

                if TEMPLATE_RE.is_match(code) {
                    continue;
                }

                let (oc, cc) =
                    code.chars().fold((0u64, 0u64), |(oc, cc), c| match c {
                        '{' => (oc + 1, cc),
                        '}' => (oc, cc + 1),
                        _ => (oc, cc),
                    });
                if oc != cc || oc == 0 || cc == 0 {
                    continue;
                }

                map.entry(code).and_modify(|c| *c += 1).or_insert(1);
            }
        }

        if map.keys().len() < NUM_INSTANTS {
            continue;
        }

        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort_by_key(|(_, k)| *k);

        let candidates: CandidateVec = pairs
            .into_iter()
            .rev()
            .filter_map(|(s, _)| {
                let elm_tree = parser.parse(s, None)?;

                let root = elm_tree.root_node();
                if root.child_count() != 1 {
                    return None;
                }

                let child = root.child(0).expect("should have one child");
                if child.kind() != "function_definition" {
                    return None;
                }

                let pcount = get_param_count(child)?;
                if pcount != tpl_param_count {
                    return None;
                }

                Some(s)
            })
            .take(5)
            .collect();

        if let Ok(s) = <[&str; NUM_INSTANTS]>::try_from(candidates.as_slice()) {
            let template = template.to_string();
            let s: [String; NUM_INSTANTS] = s.map(String::from);
            tx.send((template, s)).await?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let file = fs::File::open("generations.parquet").await?;
    let stream = ParquetRecordBatchStreamBuilder::new(file).await?.build()?;

    let (tx, mut rx) = mpsc::channel(1000);

    let generate = stream.try_for_each_concurrent(16, move |record| {
        let tx = tx.clone();
        async move {
            filter_instantiations(tx, record)
                .await
                .map_err(extract_template::into_parquet_error)
        }
    });

    let consume = async move {
        let mut templates = Vec::with_capacity(5000);
        let mut instants = Vec::with_capacity(5000);
        while let Some((t, elms)) = rx.recv().await {
            templates.push(t);
            instants.push(elms);
        }
        (templates, instants)
    };

    let (generate, consume) = tokio::join!(generate, consume);
    generate?;
    let (templates, instansts) = consume;

    let templates = Arc::new(StringArray::from(templates)) as ArrayRef;

    let mut builder =
        FixedSizeListBuilder::new(StringBuilder::new(), NUM_INSTANTS as i32);
    for insts in instansts {
        for inst in insts {
            builder.values().append_value(inst);
        }
        builder.append(true);
    }
    let instants = Arc::new(builder.finish()) as ArrayRef;

    let batch = RecordBatch::try_from_iter([("template", templates), ("instants", instants)])?;

    let schema = batch.schema_ref();
    dbg!(schema);
    let file = fs::File::create("fine-tuning.parquet").await?;
    let props = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
    let mut writer = AsyncArrowWriter::try_new(file, Arc::clone(schema), Some(props))?;

    writer.write(&batch).await?;
    writer.flush().await?;
    writer.close().await?;

    Ok(())
}
