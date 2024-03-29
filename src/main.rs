use arrow::{
    array::{Array, ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use futures::TryStreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use indicatif::{ProgressBar, ProgressIterator};
use memchr::memmem::Finder;
use parquet::{
    arrow::{
        AsyncArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask,
    },
    errors::ParquetError,
};
use regex::bytes::Regex;
use tokio::{fs, sync::mpsc};

use std::{path::PathBuf, sync::Arc};

lazy_static::lazy_static! {
    static ref TPL_FINDER: Finder<'static> = Finder::new("template");
    static ref TPL_REQUIRE: Regex = Regex::new(r"^template *<[^>]+>").unwrap();
    static ref TPL_REJECT: Regex =
        Regex::new(r"struct|class|using|[^:]:[^:]|^#include|;|~").unwrap();
}

fn into_parquet_error(
    err: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> ParquetError {
    ParquetError::External(err.into())
}

async fn filter_single_file(
    tx: mpsc::Sender<Box<str>>,
    local: PathBuf,
) -> anyhow::Result<()> {
    let file = fs::File::open(local).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;

    let file_metadata = builder.metadata().file_metadata();
    let mask = ProjectionMask::roots(file_metadata.schema_descr(), [25]);
    builder
        .with_projection(mask)
        .build()?
        .try_filter_map(|record| async move {
            Ok(record.columns().first().map(Arc::clone))
        })
        .try_for_each_concurrent(4, |arr| {
            let tx = tx.clone();
            async move {
                let arr = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        into_parquet_error(anyhow::anyhow!(
                            "not a string array"
                        ))
                    })?;
                for elm in arr
                    .into_iter()
                    .flatten()
                    .filter(|s| TPL_FINDER.find(s.as_bytes()).is_some())
                {
                    tx.send(elm.into()).await.map_err(into_parquet_error)?;
                }
                Ok(())
            }
        })
        .await?;
    Ok(())
}

async fn filter_for_template(tx: mpsc::Sender<Box<str>>) -> anyhow::Result<()> {
    let repo = Repo::with_revision(
        String::from("bigcode/the-stack-dedup"),
        RepoType::Dataset,
        String::from("main"),
    );
    let repo = &Api::new()?.repo(repo);
    let info = repo.info().await?;
    let futures = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|filename| {
            filename.starts_with("data/cpp") && filename.ends_with(".parquet")
        })
        .zip(std::iter::repeat(tx))
        .map(|(filename, tx)| async move {
            let local = repo.get(&filename).await?;
            filter_single_file(tx, local).await?;
            Ok::<_, anyhow::Error>(())
        });
    futures::future::try_join_all(futures).await?;
    Ok(())
}

async fn extract_templates(
    tx: mpsc::Sender<Box<str>>,
    code: Box<str>,
) -> anyhow::Result<()> {
    let code_bytes = code.as_bytes();
    'outer: for tpl in TPL_FINDER.find_iter(code_bytes) {
        // Make sure previous byte is not part of an identifier
        if tpl != 0
            && unicode_ident::is_xid_continue(code_bytes[tpl - 1] as char)
        {
            continue;
        }

        let code_tail = &code_bytes[tpl..];
        // Extract out the `template <...>` part if valid
        let Some(template_snip) = TPL_REQUIRE.find(code_tail) else {
            continue;
        };
        // Ensure template was found at the start of the code
        assert_eq!(template_snip.start(), 0);

        // Enumerate brackets for finding start and end
        let mut citer = code_tail.iter().cloned().enumerate().filter_map(
            |(idx, b)| match b {
                b'{' => Some((idx, true)),
                b'}' => Some((idx, false)),
                _ => None,
            },
        );

        // Get the location of first bracket
        let Some((idx, _)) = citer.next() else {
            continue;
        };

        // Extract snippet between `template <...>` and open bracket
        let Some(snip) = code_tail.get(template_snip.len()..idx) else {
            continue;
        };

        // Reject classes, structs, using definitions, semicolons, constructors,
        // and destructors
        if TPL_REJECT.find(snip).is_some() {
            continue;
        }

        // Find the end of the function body
        let mut count = 1u64;
        let end = loop {
            let Some((idx, b)) = citer.next() else {
                continue 'outer;
            };

            if b {
                count += 1;
                continue;
            }

            count -= 1;
            if count == 0 {
                break idx + 1;
            }
        };

        // Extract the template function
        let snip = unsafe { std::str::from_utf8_unchecked(&code_tail[..end]) };

        // Filter out long lines
        let mut count = 0u64;
        for l in snip.lines() {
            if l.len() > 120 {
                continue 'outer;
            }
            count += 1;
        }

        // Filter out too many lines
        if count > 40 {
            continue;
        }

        // Send off templated function
        tx.send(Box::from(snip)).await?;
    }
    Ok(())
}

async fn write_out_file(
    s: StringArray,
    mut cur_file: AsyncArrowWriter<fs::File>,
) -> anyhow::Result<()> {
    let s: ArrayRef = Arc::new(s);
    let batch = RecordBatch::try_from_iter([("template", s)])?;
    cur_file.write(&batch).await?;
    cur_file.flush().await?;
    cur_file.close().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (code_tx, mut code_rx) = mpsc::channel(1000);
    tokio::spawn(filter_for_template(code_tx));

    let (tpl_tx, mut tpl_rx) = mpsc::channel(1000);
    tokio::spawn(async move {
        while let Some(code) = code_rx.recv().await {
            tokio::spawn(extract_templates(tpl_tx.clone(), code));
        }
    });

    let schema = &Arc::new(Schema::new(vec![Field::new(
        "template",
        DataType::Utf8,
        false,
    )]));
    let mut make_new_file = (0..100_000).map(|file_count| async move {
        let f = fs::File::create(format!("{:05}.parquet", file_count)).await?;
        let w = AsyncArrowWriter::try_new(f, Arc::clone(schema), None)?;
        Ok::<_, anyhow::Error>(w)
    });

    const ROW_MAX: usize = 0x8000;
    let mut buffer = Vec::with_capacity(ROW_MAX);
    let mut file_writes = Vec::new();
    let bar = ProgressBar::new_spinner();
    while let Some(tpl_str) = tpl_rx.recv().await {
        bar.tick();
        buffer.push(tpl_str);
        if buffer.len() == ROW_MAX {
            let cur_file = make_new_file
                .next()
                .ok_or_else(|| anyhow::anyhow!("ran out of numbers for files"))?
                .await?;
            let s: StringArray =
                buffer.iter().map(|s| Some(String::from(&**s))).collect();
            buffer.clear();
            file_writes.push(tokio::spawn(write_out_file(s, cur_file)));
        }
    }
    if !buffer.is_empty() {
        let cur_file = make_new_file
            .next()
            .ok_or_else(|| anyhow::anyhow!("ran out of numbers for files"))?
            .await?;
        let s: StringArray =
            buffer.iter().map(|s| Some(String::from(&**s))).collect();
        buffer.clear();
        file_writes.push(tokio::spawn(write_out_file(s, cur_file)));
    }
    bar.finish();

    futures::future::try_join_all(file_writes)
        .await?
        .into_iter()
        .progress()
        .try_for_each(|f| f)?;

    Ok(())
}
