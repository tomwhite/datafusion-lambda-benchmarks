//! An invocation's partitions as sent to its worker. Consecutive ranges of
//! one width along the outer axis are sent as one: with a chunk a partition,
//! an invocation of hourly chunks would otherwise send a spec an hour, and a
//! fan-out's request to its coordinator grew with its chunks, not its
//! invocations.

use serde::{Deserialize, Serialize};
use zarr_datafusion::reader::filter::CoordSelection;

use crate::PartitionSpec;

/// A partition, or with `step`, the ranges of that width that tile its outer
/// range, each with its `extra`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactSpec {
    #[serde(flatten)]
    spec: PartitionSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    step: Option<usize>,
}

impl CompactSpec {
    /// Its outer range and the width of each partition in it.
    fn run(&self) -> Option<(usize, usize, usize)> {
        match self.spec.outer {
            CoordSelection::Range(start, end) => {
                Some((start, end, self.step.unwrap_or(end - start)))
            }
            CoordSelection::Indices(_) => None,
        }
    }
}

/// `specs`, each run of consecutive same-width ranges with the same `extra`
/// as one.
pub fn compact(specs: &[PartitionSpec]) -> Vec<CompactSpec> {
    let mut compacted: Vec<CompactSpec> = vec![];
    for spec in specs {
        if let (Some(last), CoordSelection::Range(start, end)) = (compacted.last_mut(), &spec.outer)
        {
            if let Some((run_start, run_end, width)) = last.run() {
                if run_end == *start
                    && end - start == width
                    && width > 0
                    && last.spec.extra == spec.extra
                {
                    last.spec.outer = CoordSelection::Range(run_start, *end);
                    last.step = Some(width);
                    continue;
                }
            }
        }
        compacted.push(CompactSpec {
            spec: spec.clone(),
            step: None,
        });
    }
    compacted
}

/// The partitions `compacted` stands for.
pub fn expand(compacted: Vec<CompactSpec>) -> Result<Vec<PartitionSpec>, String> {
    let mut specs = vec![];
    for CompactSpec { spec, step } in compacted {
        let Some(step) = step else {
            specs.push(spec);
            continue;
        };
        let CoordSelection::Range(start, end) = spec.outer else {
            return Err("a step needs a range".into());
        };
        if step == 0 || (end - start) % step != 0 {
            return Err(format!(
                "{start}..{end} isn't a whole number of steps of {step}"
            ));
        }
        specs.extend((start..end).step_by(step).map(|s| PartitionSpec {
            outer: CoordSelection::Range(s, s + step),
            extra: spec.extra.clone(),
        }));
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(bounds: &[(usize, usize)]) -> Vec<PartitionSpec> {
        bounds
            .iter()
            .map(|&(s, e)| PartitionSpec::range(s as u64, e as u64))
            .collect()
    }

    #[test]
    fn consecutive_chunks_are_sent_as_one() {
        let specs = ranges(&[(24, 25), (25, 26), (26, 27)]);
        let compacted = compact(&specs);

        assert_eq!(
            serde_json::to_string(&compacted).unwrap(),
            r#"[{"outer":{"Range":[24,27]},"extra":[],"step":1}]"#
        );
        assert_eq!(expand(compacted).unwrap(), specs);
    }

    #[test]
    fn a_partition_alone_is_sent_as_it_was() {
        let specs = ranges(&[(0, 31)]);
        let compacted = compact(&specs);

        assert_eq!(
            serde_json::to_value(&compacted).unwrap(),
            serde_json::to_value(&specs).unwrap()
        );
        assert_eq!(expand(compacted).unwrap(), specs);
    }

    #[test]
    fn gaps_widths_selections_and_extras_break_a_run() {
        let mut specs = ranges(&[(0, 2), (2, 4), (6, 8), (8, 9), (9, 10)]);
        specs.push(PartitionSpec::from_outer(CoordSelection::Indices(vec![
            11, 13,
        ])));
        specs.extend(ranges(&[(14, 15)]));
        specs.push(PartitionSpec {
            outer: CoordSelection::Range(15, 16),
            extra: vec![(1, CoordSelection::Range(0, 4))],
        });
        let compacted = compact(&specs);

        assert_eq!(compacted.len(), 6);
        assert_eq!(expand(compacted).unwrap(), specs);
    }

    #[test]
    fn a_ragged_run_is_refused() {
        let ragged = CompactSpec {
            spec: PartitionSpec::range(0, 5),
            step: Some(2),
        };
        assert!(expand(vec![ragged]).is_err());
    }
}
