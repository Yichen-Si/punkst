#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace punkst {

template <typename Value>
class LbfgsHistory {
public:
    explicit LbfgsHistory(int32_t capacity) : capacity_(capacity) {}

    Value apply(const Value& gradient) const {
        Value direction = gradient;
        std::vector<double> alpha(s_.size());
        for (size_t reverse = s_.size(); reverse > 0; --reverse) {
            const size_t index = reverse - 1;
            alpha[index] = rho_[index] * inner(s_[index], direction);
            direction.noalias() -= alpha[index] * y_[index];
        }
        if (!s_.empty()) {
            const double denominator = inner(y_.back(), y_.back());
            if (denominator > 0.0) {
                direction *= inner(s_.back(), y_.back()) / denominator;
            }
        }
        for (size_t index = 0; index < s_.size(); ++index) {
            const double beta = rho_[index] * inner(y_[index], direction);
            direction.noalias() += s_[index] * (alpha[index] - beta);
        }
        return direction;
    }

    bool update(Value step, Value gradient_change) {
        const double curvature = inner(step, gradient_change);
        const double threshold = 1e-12
            * std::sqrt(inner(step, step) * inner(gradient_change, gradient_change));
        if (!std::isfinite(curvature) || !(curvature > threshold)) return false;
        if (static_cast<int32_t>(s_.size()) == capacity_) {
            s_.erase(s_.begin());
            y_.erase(y_.begin());
            rho_.erase(rho_.begin());
        }
        s_.push_back(std::move(step));
        y_.push_back(std::move(gradient_change));
        rho_.push_back(1.0 / curvature);
        return true;
    }

    void clear() {
        s_.clear();
        y_.clear();
        rho_.clear();
    }

    bool empty() const { return s_.empty(); }
    size_t size() const { return s_.size(); }

private:
    static double inner(const Value& left, const Value& right) {
        return (left.array() * right.array()).sum();
    }

    int32_t capacity_;
    std::vector<Value> s_;
    std::vector<Value> y_;
    std::vector<double> rho_;
};

} // namespace punkst
