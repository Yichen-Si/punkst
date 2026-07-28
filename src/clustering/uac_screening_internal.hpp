#pragma once

#include "clustering/uac_common_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace uac::detail {

struct ScreenedComponents {
    Eigen::VectorXd score;
    std::vector<int32_t> evaluated;
    double log_mass = -std::numeric_limits<double>::infinity();
    double log_upper_mass = -std::numeric_limits<double>::infinity();
    double omitted_mass_bound = 0.0;
    bool full = true;
    bool bound_violation = false;
};

struct ComponentScreeningWorkspace {
    std::vector<int32_t> order;
    std::vector<double> suffix;
};

template<class Evaluate>
ScreenedComponents screen_component_scores(
    const Eigen::Ref<const Eigen::VectorXd>& upper,
    const ComponentScreeningOptions& options, bool enabled,
    Evaluate&& evaluate, ComponentScreeningWorkspace* supplied = nullptr) {
    const int32_t components = static_cast<int32_t>(upper.size());
    ScreenedComponents out;
    out.score = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    out.evaluated.reserve(components);
    ComponentScreeningWorkspace local;
    ComponentScreeningWorkspace& workspace =
        supplied == nullptr ? local : *supplied;
    std::vector<int32_t>& order = workspace.order;
    order.clear();
    order.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        if (std::isfinite(upper(c))) order.push_back(c);
    }
    if (order.empty()) {
        throw std::runtime_error("UAC component screen has no active component");
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t left,
            int32_t right) {
        return upper(left) == upper(right)
            ? left < right : upper(left) > upper(right);
    });
    std::vector<double>& suffix = workspace.suffix;
    suffix.assign(order.size() + 1,
        -std::numeric_limits<double>::infinity());
    for (size_t i = order.size(); i > 0; --i) {
        suffix[i - 1] = logaddexp(upper(order[i - 1]), suffix[i]);
    }
    const int32_t minimum = enabled
        ? std::min<int32_t>(options.minimum_components, order.size())
        : static_cast<int32_t>(order.size());
    const int32_t maximum =
        enabled && options.mode == ComponentScreeningMode::On
            && options.maximum_components > 0
        ? std::min<int32_t>(options.maximum_components, order.size())
        : static_cast<int32_t>(order.size());
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const int32_t component = order[rank];
        const double exact = evaluate(component);
        out.score(component) = exact;
        out.evaluated.push_back(component);
        out.log_mass = logaddexp(out.log_mass, exact);
        const double tolerance = 1e-10
            * std::max(1.0, std::abs(upper(component)));
        if (exact > upper(component) + tolerance) {
            out.bound_violation = true;
            enabled = false;
        }
        const bool reached_maximum =
            static_cast<int32_t>(out.evaluated.size()) >= maximum;
        if (reached_maximum) {
            out.log_upper_mass = suffix[rank + 1];
            const double combined =
                logaddexp(out.log_mass, out.log_upper_mass);
            out.omitted_mass_bound = std::isfinite(out.log_upper_mass)
                ? std::exp(out.log_upper_mass - combined) : 0.0;
            if (out.bound_violation && rank + 1 < order.size()) {
                out.omitted_mass_bound = 1.0;
            }
            break;
        }
        if (!enabled
            || static_cast<int32_t>(out.evaluated.size()) < minimum) {
            continue;
        }
        out.log_upper_mass = suffix[rank + 1];
        const double combined =
            logaddexp(out.log_mass, out.log_upper_mass);
        out.omitted_mass_bound = std::isfinite(out.log_upper_mass)
            ? std::exp(out.log_upper_mass - combined) : 0.0;
        if (out.omitted_mass_bound <= options.tail_mass) break;
    }
    const bool forced_maximum =
        options.mode == ComponentScreeningMode::On
        && options.maximum_components > 0;
    if (!enabled && !forced_maximum
        && out.evaluated.size() < order.size()) {
        for (size_t rank = out.evaluated.size(); rank < order.size(); ++rank) {
            const int32_t component = order[rank];
            const double exact = evaluate(component);
            out.score(component) = exact;
            out.evaluated.push_back(component);
            out.log_mass = logaddexp(out.log_mass, exact);
        }
        out.log_upper_mass = -std::numeric_limits<double>::infinity();
        out.omitted_mass_bound = 0.0;
    }
    out.full = out.evaluated.size() == order.size();
    if (out.full) {
        out.log_upper_mass = -std::numeric_limits<double>::infinity();
        out.omitted_mass_bound = 0.0;
    }
    return out;
}

} // namespace uac::detail
