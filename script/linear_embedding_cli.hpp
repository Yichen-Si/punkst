#pragma once

#include "commands.hpp"
#include "linear_embedding.hpp"

#include <stdexcept>
#include <string>

namespace punkst_cli {

struct LinearEmbeddingCliOptions {
    punkst::linear_embedding::Options values;
    std::string projection_space = "linear";
    std::string whitening = "mixture";
    bool include_full = false;
    bool skip_qda_projection = false;
    bool qda_sparsity_cv = false;

    void add_qda_options(ParamList& parameters) {
        parameters
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
              values.qda_sparsity_grid);
    }

    void finalize_qda_options(const ParamList& parameters) {
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
    }
};

} // namespace punkst_cli
