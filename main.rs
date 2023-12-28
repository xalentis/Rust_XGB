// Gideon Vos 2023
// Example implementation of a stress prediction model in Rust using a synthesized stress biomarker dataset available here:
// https://github.com/xalentis/Stress
// uses xgboost with LOSO validation

use polars::prelude::*;
use std::collections::HashSet;
use xgboost::{parameters, DMatrix, Booster};

// round a vector
fn round_to_one(numbers: &mut Vec<f32>) {
    for num in numbers.iter_mut() {
        *num = num.round();
    }
}

// calculate a binary accuracy score
fn accuracy_score<T: PartialEq>(vector1: &[T], vector2: &[T]) -> f32 {
    let equal_count = vector1.iter().zip(vector2.iter()).filter(|(elem1, elem2)| elem1 == elem2).count() as f32;
    let total_elements = vector1.len() as f32;
    (equal_count / total_elements) * 100.0
}

// merge a number of polars vectors column-wise
fn combine_column_wise(vectors: Vec<Vec<f32>>) -> Vec<f32> {
    let len = vectors[0].len();
    assert!(vectors.iter().all(|vec| vec.len() == len), "All vectors must have the same length");
    let mut combined: Vec<f32> = Vec::with_capacity(len * vectors.len());
    for col in 0..len {
        for vec in &vectors {
            combined.push(vec[col]);
        }
    }
    combined
}


fn main() -> PolarsResult<()> {
    let schema = Schema::from(vec![
            Field::new("hrrange", DataType::Float32),
            Field::new("hrvar", DataType::Float32),
            Field::new("hrstd", DataType::Float32),
            Field::new("hrmin", DataType::Float32),
            Field::new("edarange", DataType::Float32),
            Field::new("edastd", DataType::Float32),
            Field::new("edavar", DataType::Float32),
            Field::new("hrkurt", DataType::Float32),
            Field::new("edamin", DataType::Float32),
            Field::new("hrmax", DataType::Float32),
            Field::new("Subject", DataType::Utf8),
            Field::new("metric", DataType::Float32),
        ].into_iter());

    // load dataset from csv
    let df = CsvReader::from_path("/data/Projects/Rust/stressXGB/SynthesizedStressData.csv")?
        .with_schema(&schema)
        .has_header(true)
        .finish()
        .unwrap();

    let binding = df.clone();
    let subjects:Vec<&str> = binding["Subject"].utf8().unwrap().into_no_null_iter().collect();
    let unique_subjects: HashSet<&str> = subjects.into_iter().collect();
    let subjects:Vec<&str> = unique_subjects.into_iter().collect();

    // LOSO training/eval loop
    for subject in &subjects {
        let filter_expr = col("Subject").eq(lit(String::from(*subject)));
        let df_test = df.clone().lazy().filter(filter_expr).collect().unwrap();
        let filter_expr = col("Subject").neq(lit(String::from(*subject)));
        let df_train = df.clone().lazy().filter(filter_expr).collect().unwrap();

        // make training dataset
        let y_train:Vec<f32> = df_train.clone()["metric"].f32().unwrap().into_no_null_iter().collect();
        let hrrange_train: Vec<f32> = df_train.clone()["hrrange"].f32().unwrap().into_no_null_iter().collect();
        let hrvar_train: Vec<f32> = df_train.clone()["hrvar"].f32().unwrap().into_no_null_iter().collect();
        let hrstd_train: Vec<f32> = df_train.clone()["hrstd"].f32().unwrap().into_no_null_iter().collect();
        let hrmin_train: Vec<f32> = df_train.clone()["hrmin"].f32().unwrap().into_no_null_iter().collect();
        let edarange_train: Vec<f32> = df_train.clone()["edarange"].f32().unwrap().into_no_null_iter().collect();
        let edastd_train: Vec<f32> = df_train.clone()["edastd"].f32().unwrap().into_no_null_iter().collect();
        let edavar_train: Vec<f32> = df_train.clone()["edavar"].f32().unwrap().into_no_null_iter().collect();
        let hrkurt_train: Vec<f32> = df_train.clone()["hrkurt"].f32().unwrap().into_no_null_iter().collect();
        let edamin_train: Vec<f32> = df_train.clone()["edamin"].f32().unwrap().into_no_null_iter().collect();
        let hrmax_train: Vec<f32> = df_train.clone()["hrmax"].f32().unwrap().into_no_null_iter().collect();
        let x_train: Vec<f32> = combine_column_wise(vec![hrrange_train, hrvar_train, hrstd_train, hrmin_train, edarange_train, edastd_train, edavar_train, hrkurt_train, edamin_train, hrmax_train]);
        let num_rows = df_train.height();
        let mut dtrain = DMatrix::from_dense(&x_train, num_rows).unwrap();
        dtrain.set_labels(&y_train).unwrap();

        // make test dataset of single hold-out subject not in training dataset
        let y_test:Vec<f32> = df_test.clone()["metric"].f32().unwrap().into_no_null_iter().collect();
        let hrrange_test: Vec<f32> = df_test.clone()["hrrange"].f32().unwrap().into_no_null_iter().collect();
        let hrvar_test: Vec<f32> = df_test.clone()["hrvar"].f32().unwrap().into_no_null_iter().collect();
        let hrstd_test: Vec<f32> = df_test.clone()["hrstd"].f32().unwrap().into_no_null_iter().collect();
        let hrmin_test: Vec<f32> = df_test.clone()["hrmin"].f32().unwrap().into_no_null_iter().collect();
        let edarange_test: Vec<f32> = df_test.clone()["edarange"].f32().unwrap().into_no_null_iter().collect();
        let edastd_test: Vec<f32> = df_test.clone()["edastd"].f32().unwrap().into_no_null_iter().collect();
        let edavar_test: Vec<f32> = df_test.clone()["edavar"].f32().unwrap().into_no_null_iter().collect();
        let hrkurt_test: Vec<f32> = df_test.clone()["hrkurt"].f32().unwrap().into_no_null_iter().collect();
        let edamin_test: Vec<f32> = df_test.clone()["edamin"].f32().unwrap().into_no_null_iter().collect();
        let hrmax_test: Vec<f32> = df_test.clone()["hrmax"].f32().unwrap().into_no_null_iter().collect();
        let x_test: Vec<f32> = combine_column_wise(vec![hrrange_test, hrvar_test, hrstd_test, hrmin_test, edarange_test, edastd_test, edavar_test, hrkurt_test, edamin_test, hrmax_test]);
        let num_rows = df_test.height();
        let mut dtest = DMatrix::from_dense(&x_test, num_rows).unwrap();
        dtest.set_labels(&y_test).unwrap();

        // xgboost configuration
        let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::RegLogistic)
        .eval_metrics(parameters::learning::Metrics::Custom(vec![parameters::learning::EvaluationMetric::RMSE]))
        .seed(42)
        .build().unwrap();

        let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .updater(vec![parameters::tree::TreeUpdater::GrowColMaker, parameters::tree::TreeUpdater::Prune])
            .max_depth(8)
            .eta(0.5)
            .subsample(0.7)
            .colsample_bytree(0.8)
            .build().unwrap();

        let booster_params = parameters::BoosterParametersBuilder::default()
            .learning_params(learning_params)
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .verbose(false)
            .build().unwrap();

        let params = parameters::TrainingParametersBuilder::default()
            .dtrain(&dtrain)
            .boost_rounds(200)
            .booster_params(booster_params)
            .build().unwrap();

        // train model, and print evaluation data
        let bst = Booster::train(&params).unwrap();
        let mut preds = bst.predict(&dtest).unwrap();
        round_to_one(&mut preds);
        let acc = accuracy_score(&preds, &y_test);
        println!("Subject {} - Accuracy: {}", subject, acc);
    }

    Ok(())
}