//! Evaluating an expression of one dictionary-encoded column once per
//! distinct value rather than once per row. Zarr coordinates arrive as
//! dictionaries over a chunk's coordinate values, but a cast or a function
//! over one unpacks it, so `date_trunc('month', time)` would otherwise run
//! once for every cell.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr_common::physical_expr::is_volatile;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, PhysicalGroupBy};
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::ExecutionPlan;

/// Rewrites projections, filters and aggregates so that each deterministic
/// expression of a single dictionary-encoded column runs over the
/// dictionary's values. Runs last: the rewritten expressions are opaque to
/// other rules, and don't serialize.
#[derive(Debug, Default)]
pub struct PerValueRule;

impl PhysicalOptimizerRule for PerValueRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        rewrite_plan(plan)
    }

    fn name(&self) -> &str {
        "per_value"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

pub fn rewrite_plan(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(plan.transform_up(rewrite_node)?.data)
}

fn rewrite_node(node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let rewritten: Option<Arc<dyn ExecutionPlan>> =
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            let schema = projection.input().schema();
            let mut changed = false;
            let exprs = projection
                .expr()
                .iter()
                .map(|e| {
                    let t = rewrite(&e.expr, &schema)?;
                    changed |= t.transformed;
                    Ok(ProjectionExpr::new(t.data, e.alias.clone()))
                })
                .collect::<Result<Vec<_>>>()?;
            changed
                .then(|| ProjectionExec::try_new(exprs, projection.input().clone()))
                .transpose()?
                .map(|p| Arc::new(p) as _)
        } else if let Some(filter) = node.downcast_ref::<FilterExec>() {
            let t = rewrite(filter.predicate(), &filter.input().schema())?;
            t.transformed
                .then(|| {
                    FilterExecBuilder::from(filter)
                        .with_predicate(t.data)
                        .build()
                })
                .transpose()?
                .map(|f| Arc::new(f) as _)
        } else if let Some(aggregate) = node.downcast_ref::<AggregateExec>() {
            rewrite_aggregate(aggregate)?
        } else {
            None
        };
    Ok(match rewritten {
        // a rewrite must not change what the node produces
        Some(new) if new.schema() == node.schema() => Transformed::yes(new),
        _ => Transformed::no(node),
    })
}

fn rewrite_aggregate(aggregate: &AggregateExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let schema = aggregate.input().schema();
    let mut changed = false;
    let mut rewrite = |e: &Arc<dyn PhysicalExpr>| -> Result<Arc<dyn PhysicalExpr>> {
        let t = rewrite(e, &schema)?;
        changed |= t.transformed;
        Ok(t.data)
    };
    let group_by = aggregate.group_expr();
    let mut named = |exprs: &[(Arc<dyn PhysicalExpr>, String)]| {
        exprs
            .iter()
            .map(|(e, name)| Ok((rewrite(e)?, name.clone())))
            .collect::<Result<Vec<_>>>()
    };
    let expr = named(group_by.expr())?;
    let null_expr = named(group_by.null_expr())?;
    let group_by = PhysicalGroupBy::new(
        expr,
        null_expr,
        group_by.groups().to_vec(),
        group_by.has_grouping_set(),
    );
    let mut aggr_expr = vec![];
    for a in aggregate.aggr_expr() {
        let all = a.all_expressions();
        let args = all.args.iter().map(&mut rewrite).collect::<Result<_>>()?;
        match a.with_new_expressions(args, all.order_by_exprs) {
            Some(new) => aggr_expr.push(Arc::new(new)),
            None => return Ok(None),
        }
    }
    let filter_expr = aggregate
        .filter_expr()
        .iter()
        .map(|f| f.as_ref().map(&mut rewrite).transpose())
        .collect::<Result<Vec<_>>>()?;
    if !changed {
        return Ok(None);
    }
    let new = AggregateExec::try_new(
        *aggregate.mode(),
        group_by,
        aggr_expr,
        filter_expr,
        aggregate.input().clone(),
        aggregate.input_schema(),
    )?
    .with_new_limit_options(aggregate.limit_options());
    Ok(Some(Arc::new(new)))
}

/// `expr` with each largest subexpression that can run per value replaced.
fn rewrite(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    expr.clone().transform_down(|e| {
        if e.downcast_ref::<PerValue>().is_some() {
            return Ok(Transformed::new(e, false, TreeNodeRecursion::Jump));
        }
        Ok(match PerValue::try_new(&e, schema) {
            Some(p) => Transformed::new(Arc::new(p), true, TreeNodeRecursion::Jump),
            None => Transformed::no(e),
        })
    })
}

/// `original`, evaluated over the values of its one column's dictionary and
/// gathered by the keys. Falls back to `original` whenever that could differ:
/// for null keys, which `original` may map to non-null results, when there
/// are no fewer values than rows, and when evaluating the values fails, since
/// they can include values no row refers to.
#[derive(Debug)]
pub struct PerValue {
    original: Arc<dyn PhysicalExpr>,
    /// `original` over a batch of the values alone.
    on_values: Arc<dyn PhysicalExpr>,
    column: usize,
    values_field: FieldRef,
}

impl PerValue {
    fn try_new(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<Self> {
        if expr.downcast_ref::<Column>().is_some() || is_volatile(expr) {
            return None;
        }
        let columns = collect_columns(expr);
        let [column] = columns.iter().collect::<Vec<_>>()[..] else {
            return None;
        };
        let field = schema.fields().get(column.index())?;
        let DataType::Dictionary(_, value_type) = field.data_type() else {
            return None;
        };
        let values_field = Arc::new(Field::new(
            field.name(),
            value_type.as_ref().clone(),
            field.is_nullable(),
        ));
        let on_values = expr
            .clone()
            .transform(|e| {
                Ok(match e.downcast_ref::<Column>() {
                    Some(c) if c.index() == column.index() => {
                        Transformed::yes(Arc::new(Column::new(c.name(), 0)) as _)
                    }
                    _ => Transformed::no(e),
                })
            })
            .ok()?
            .data;
        let values_schema = Schema::new(vec![values_field.clone()]);
        if on_values.data_type(&values_schema).ok()? != expr.data_type(schema).ok()? {
            return None;
        }
        Some(Self {
            original: expr.clone(),
            on_values,
            column: column.index(),
            values_field,
        })
    }

    fn per_value(&self, batch: &RecordBatch) -> Option<ColumnarValue> {
        let dictionary = batch.column(self.column).as_any_dictionary_opt()?;
        let values = dictionary.values();
        if dictionary.keys().null_count() > 0 || values.len() >= batch.num_rows() {
            return None;
        }
        let values_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![self.values_field.clone()])),
            vec![values.clone()],
        )
        .ok()?;
        let mapped = self
            .on_values
            .evaluate(&values_batch)
            .and_then(|v| v.into_array(values.len()))
            .ok()?;
        take(&mapped, dictionary.keys(), None)
            .ok()
            .map(ColumnarValue::Array)
    }
}

impl PhysicalExpr for PerValue {
    fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> {
        self.original.return_field(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        match self.per_value(batch) {
            Some(result) => Ok(result),
            None => self.original.evaluate(batch),
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.original]
    }

    /// Any change below drops back to the plain expression, since the per
    /// value form needs the input schema to rebuild.
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(children.remove(0))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.original.fmt_sql(f)
    }
}

// `on_values` and `values_field` follow from the rest.
impl PartialEq for PerValue {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column && self.original.as_ref() == other.original.as_ref()
    }
}

impl Eq for PerValue {}

impl Hash for PerValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.original.as_ref().hash(state);
        self.column.hash(state);
    }
}

impl fmt::Display for PerValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "per_value({})", self.original)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{DictionaryArray, Float64Array, Int16Array, Int64Array};
    use arrow::datatypes::Int16Type;
    use datafusion::datasource::MemTable;
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, cast, is_null, lit};
    use datafusion::prelude::SessionContext;

    use super::*;

    /// `d`, a dictionary of three values over `keys`, next to `n`.
    fn batch(keys: Vec<Option<i16>>) -> RecordBatch {
        let n = Int64Array::from_iter_values(0..keys.len() as i64);
        let d = DictionaryArray::<Int16Type>::new(
            Int16Array::from(keys),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        );
        RecordBatch::try_from_iter([("n", Arc::new(n) as _), ("d", Arc::new(d) as _)]).unwrap()
    }

    fn per_value(expr: Arc<dyn PhysicalExpr>, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        let t = rewrite(&expr, schema).unwrap();
        assert!(t.transformed, "{expr} wasn't rewritten");
        t.data
    }

    fn evaluate(expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> arrow::array::ArrayData {
        expr.evaluate(batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap()
            .to_data()
    }

    #[test]
    fn matches_the_expression_it_replaces() {
        let batch = batch((0..10).map(|k| Some(k % 3)).collect());
        let schema = batch.schema();
        let d = cast(Arc::new(Column::new("d", 1)), &schema, DataType::Float64).unwrap();
        let expr = binary(d, Operator::Multiply, lit(2.0), &schema).unwrap();

        let rewritten = per_value(expr.clone(), &schema);

        assert!(rewritten.downcast_ref::<PerValue>().is_some());
        assert_eq!(evaluate(&rewritten, &batch), evaluate(&expr, &batch));
    }

    #[test]
    fn falls_back_for_null_keys() {
        let batch = batch(vec![Some(0), None, Some(1), Some(2), None]);
        let schema = batch.schema();
        let expr = is_null(Arc::new(Column::new("d", 1))).unwrap();

        let rewritten = per_value(expr.clone(), &schema);

        assert_eq!(evaluate(&rewritten, &batch), evaluate(&expr, &batch));
    }

    #[test]
    fn leaves_expressions_of_other_columns() {
        let schema = batch(vec![Some(0)]).schema();
        let n = Arc::new(Column::new("n", 0));
        let d = cast(Arc::new(Column::new("d", 1)), &schema, DataType::Int64).unwrap();
        let both = binary(n.clone(), Operator::Plus, d, &schema).unwrap();
        let plain = binary(n, Operator::Plus, lit(1i64), &schema).unwrap();

        let t = rewrite(&both, &schema).unwrap();
        // only the cast of `d` inside it
        assert!(t.transformed);
        assert!(t.data.downcast_ref::<PerValue>().is_none());
        assert!(!rewrite(&plain, &schema).unwrap().transformed);
    }

    async fn sql(rule: bool, query: &str) -> Vec<RecordBatch> {
        let mut builder = SessionStateBuilder::new().with_default_features();
        if rule {
            builder = builder.with_physical_optimizer_rule(Arc::new(PerValueRule));
        }
        let ctx = SessionContext::new_with_state(builder.build());
        let batch = batch((0..100).map(|k| Some(k % 3)).collect());
        let table = MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx.sql(query).await.unwrap().collect().await.unwrap()
    }

    #[tokio::test]
    async fn a_query_gives_the_same_answer_with_the_rule() {
        let query = "SELECT CAST(d AS BIGINT) * 10 AS g, sum(n) AS s, sum(sqrt(d)) AS r \
                     FROM t WHERE d + 1 > 2 GROUP BY 1 ORDER BY 1";
        let plan = sql(true, &format!("EXPLAIN {query}")).await;
        let plan = arrow::util::pretty::pretty_format_batches(&plan)
            .unwrap()
            .to_string();

        assert!(plan.contains("per_value("), "{plan}");
        assert_eq!(sql(true, query).await, sql(false, query).await);
    }
}
