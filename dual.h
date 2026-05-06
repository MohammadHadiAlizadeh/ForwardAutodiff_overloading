#ifndef DUAL_H
#define DUAL_H


#include <cmath>

#include <vector>
#include <stdexcept>


// ============================================================
// Dual Number Class (Forward Mode AD)
// ============================================================

template<typename T>
class Dual {
private:
    T val_;
    std::vector<T> deriv_;

public:
    // Constant
    Dual(T v = 0, int nvars = 0)
        : val_(v), deriv_(nvars, 0) {}

    // Variable (derivative = 1 at index)
    Dual(T v, int idx, int nvars)
        : val_(v), deriv_(nvars, 0) {
        deriv_[idx] = 1;
    }

    // Full constructor
    Dual(T v, const std::vector<T>& d)
        : val_(v), deriv_(d) {}

    T value() const { return val_; }
    const std::vector<T>& derivative() const { return deriv_; }

    // Safety check
    void check_size(const Dual& other) const {
        if (deriv_.size() != other.deriv_.size())
            throw std::runtime_error("Derivative size mismatch");
    }

    // Arithmetic operators
    Dual operator+(const Dual& other) const {
        check_size(other);
        std::vector<T> d(deriv_.size());

        for (size_t i = 0; i < d.size(); ++i)
            d[i] = deriv_[i] + other.deriv_[i];

        return {val_ + other.val_, d};
    }

    Dual operator-(const Dual& other) const {
        check_size(other);
        std::vector<T> d(deriv_.size());

        for (size_t i = 0; i < d.size(); ++i)
            d[i] = deriv_[i] - other.deriv_[i];

        return {val_ - other.val_, d};
    }

    Dual operator*(const Dual& other) const {
        check_size(other);
        std::vector<T> d(deriv_.size());

        for (size_t i = 0; i < d.size(); ++i)
            d[i] = deriv_[i] * other.val_ + val_ * other.deriv_[i];

        return {val_ * other.val_, d};
    }

    Dual operator/(const Dual& other) const {
        check_size(other);
        std::vector<T> d(deriv_.size());

        T denom = other.val_ * other.val_;
        for (size_t i = 0; i < d.size(); ++i)
            d[i] = (deriv_[i] * other.val_ - val_ * other.deriv_[i]) / denom;

        return {val_ / other.val_, d};
    }

    Dual operator-() const {
        std::vector<T> d(deriv_.size());

        for (size_t i = 0; i < d.size(); ++i)
            d[i] = -deriv_[i];

        return {-val_, d};
    }
};
#endif // DUAL_H
