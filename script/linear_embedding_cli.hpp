#pragma once

#include "commands.hpp"
#include "error.hpp"
#include "linear_embedding.hpp"

#include <stdexcept>
#include <string>

namespace punkst_cli {

struct LinearEmbeddingCliOptions {
    punkst::linear_embedding::Options values;
    std::string projection_space = "linear";
    std::string whitening = "mixture";
    bool include_full = false;
    bool skip_eigen_projection = false;
    bool skip_qda_projection = false;
    bool qda_sparsity_cv = false;
    bool skip_lda_projection = false;
    bool lda_sparsity_cv = false;

    void add_discriminant_options(ParamList& parameters) {
        parameters
          .add_option("min-cover-mass",
              "Retain decreasing-mass factors through this cumulative mass proportion; 1 disables",
              values.min_cover_mass)
          .add_option("min-mass",
              "Minimum retained factor mass proportion; 0 disables",
              values.min_mass)
          .add_option("skip-eigen-projection",
              "Skip the eigendecomposition-based projection",
              skip_eigen_projection)
          .add_option("skip-qda-projection",
              "Do not learn the default conditional-QDA projection",
              skip_qda_projection)
          .add_option("qda-train-max-rows",
              "Hard ceiling for stratified QDA training rows; 0 disables the hard ceiling",
              values.qda_train_max_rows)
          .add_option("qda-validation-max-rows",
              "Maximum stratified QDA validation rows; 0 uses all",
              values.qda_validation_max_rows)
          .add_option("qda-validation-fraction",
              "Per-class fraction reserved for QDA validation",
              values.qda_validation_fraction)
          .add_option("qda-epochs", "Maximum QDA Adam epochs",
              values.qda_epochs)
          .add_option("qda-learning-rate", "QDA Adam learning rate",
              values.qda_learning_rate)
          .add_option("qda-covariance-shrinkage",
              "QDA covariance shrinkage toward the global covariance",
              values.qda_covariance_shrinkage)
          .add_option("qda-ridge", "Positive QDA covariance ridge",
              values.qda_ridge)
          .add_option("qda-restarts", "Number of QDA optimizer restarts",
              values.qda_restarts)
          .add_option("qda-eval-every",
              "QDA validation interval in epochs",
              values.qda_evaluate_every)
          .add_option("qda-patience",
              "QDA validation checks without improvement before stopping",
              values.qda_patience)
          .add_option("qda-seed", "QDA split and optimizer seed",
              values.qda_seed)
          .add_option("qda-sparsity-strength",
              "Quartimax reward strength for QDA optimization",
              values.qda_sparsity_strength)
          .add_option("qda-sparsity-cv",
              "Select QDA sparsity strength by nested cross-validation",
              qda_sparsity_cv)
          .add_option("qda-sparsity-cv-folds",
              "Maximum stratified outer folds for QDA sparsity selection",
              values.qda_sparsity_cv_folds)
          .add_option("qda-sparsity-grid",
              "Candidate QDA sparsity strengths for cross-validation",
              values.qda_sparsity_grid)
          .add_option("skip-lda-projection",
              "Skip conditional-LDA projection",
              skip_lda_projection)
          .add_option("lda-train-max-rows",
              "Hard ceiling for stratified LDA training rows; 0 disables the hard ceiling",
              values.lda_train_max_rows)
          .add_option("lda-validation-max-rows",
              "Maximum stratified LDA validation rows; 0 uses all",
              values.lda_validation_max_rows)
          .add_option("lda-validation-fraction",
              "Per-class fraction reserved for LDA validation",
              values.lda_validation_fraction)
          .add_option("lda-epochs", "Maximum LDA Adam epochs",
              values.lda_epochs)
          .add_option("lda-learning-rate", "LDA Adam learning rate",
              values.lda_learning_rate)
          .add_option("lda-covariance-shrinkage",
              "LDA pooled-covariance shrinkage toward scaled identity",
              values.lda_covariance_shrinkage)
          .add_option("lda-ridge", "Positive LDA covariance ridge",
              values.lda_ridge)
          .add_option("lda-restarts", "Number of LDA optimizer restarts",
              values.lda_restarts)
          .add_option("lda-eval-every",
              "LDA validation interval in epochs",
              values.lda_evaluate_every)
          .add_option("lda-patience",
              "LDA validation checks without improvement before stopping",
              values.lda_patience)
          .add_option("lda-seed", "LDA split and optimizer seed",
              values.lda_seed)
          .add_option("lda-sparsity-strength",
              "Quartimax reward strength for LDA optimization",
              values.lda_sparsity_strength)
          .add_option("lda-sparsity-cv",
              "Select LDA sparsity strength by nested cross-validation",
              lda_sparsity_cv)
          .add_option("lda-sparsity-cv-folds",
              "Maximum stratified outer folds for LDA sparsity selection",
              values.lda_sparsity_cv_folds)
          .add_option("lda-sparsity-grid",
              "Candidate LDA sparsity strengths for cross-validation",
              values.lda_sparsity_grid);
    }

    void finalize_discriminant_options(const ParamList& parameters) {
        values.eigen_projection = !skip_eigen_projection;
        const bool has_strength =
            parameters.was_provided("qda-sparsity-strength");
        const bool has_grid = parameters.was_provided("qda-sparsity-grid");
        const bool has_folds =
            parameters.was_provided("qda-sparsity-cv-folds");
        if (skip_qda_projection
                && (has_strength || qda_sparsity_cv
                    || has_grid || has_folds)) {
            throw std::invalid_argument(
                "QDA sparsity options cannot be used with --skip-qda-projection");
        }
        if (qda_sparsity_cv && has_strength) {
            throw std::invalid_argument(
                "--qda-sparsity-strength and --qda-sparsity-cv are mutually exclusive");
        }
        if (!qda_sparsity_cv && (has_grid || has_folds)) {
            throw std::invalid_argument(
                "--qda-sparsity-grid and --qda-sparsity-cv-folds require --qda-sparsity-cv");
        }
        values.qda_projection = !skip_qda_projection;
        values.qda_sparsity_cv = qda_sparsity_cv;

        const bool has_lda_strength =
            parameters.was_provided("lda-sparsity-strength");
        const bool has_lda_grid = parameters.was_provided("lda-sparsity-grid");
        const bool has_lda_folds =
            parameters.was_provided("lda-sparsity-cv-folds");
        const char* const lda_tuning_options[] = {
            "lda-train-max-rows", "lda-validation-max-rows",
            "lda-validation-fraction", "lda-epochs", "lda-learning-rate",
            "lda-covariance-shrinkage", "lda-ridge", "lda-restarts",
            "lda-eval-every", "lda-patience", "lda-seed",
            "lda-sparsity-strength", "lda-sparsity-cv",
            "lda-sparsity-cv-folds", "lda-sparsity-grid",
        };
        if (skip_lda_projection) {
            bool flag = false;
            for (const char* option : lda_tuning_options) {
                flag = flag || parameters.was_provided(option);
            }
            if (flag) {
                warning("LDA options are ignored when --skip-lda-projection is specified");
            }
        } else {
            if (lda_sparsity_cv && has_lda_strength) {
                throw std::invalid_argument(
                    "--lda-sparsity-strength and --lda-sparsity-cv are mutually exclusive");
            }
            if (!lda_sparsity_cv && (has_lda_grid || has_lda_folds)) {
                throw std::invalid_argument(
                    "--lda-sparsity-grid and --lda-sparsity-cv-folds require --lda-sparsity-cv");
            }
        }
        values.lda_projection = !skip_lda_projection;
        values.lda_sparsity_cv = !skip_lda_projection && lda_sparsity_cv;
    }
};

} // namespace punkst_cli
