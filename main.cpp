#include<iostream>
#include <cmath>
#include <iomanip>
#include <memory>
#include <vector>
#include <functional>
#include "dual.h"

// ============================================================
// Dual Number Class (Forward Mode AD)
// ============================================================

// ============================================================
// Math functions (namespace to avoid conflicts)
// ============================================================
namespace ad {

template<typename T>
Dual<T> sin(const Dual<T>& x) {
    T v = std::sin(x.value());
    auto d = x.derivative();

    T c = std::cos(x.value());
    for (auto& di : d) di *= c;

    return {v, d};
}

template<typename T>
Dual<T> cos(const Dual<T>& x) {
    T v = std::cos(x.value());
    auto d = x.derivative();

    T s = -std::sin(x.value());
    for (auto& di : d) di *= s;

    return {v, d};
}

template<typename T>
Dual<T> exp(const Dual<T>& x) {
    T v = std::exp(x.value());
    auto d = x.derivative();

    for (auto& di : d) di *= v;

    return {v, d};
}

template<typename T>
Dual<T> log(const Dual<T>& x) {
    T v = std::log(x.value());
    auto d = x.derivative();

    T inv = 1.0 / x.value();
    for (auto& di : d) di *= inv;

    return {v, d};
}

template<typename T>
Dual<T> pow(const Dual<T>& x, T p) {
    T v = std::pow(x.value(), p);
    auto d = x.derivative();

    T factor = p * std::pow(x.value(), p - 1);
    for (auto& di : d) di *= factor;

    return {v, d};
}

} // namespace ad

// ============================================================
// Expression Graph
// ============================================================
class ExprNode {
public:
    virtual ~ExprNode() = default;
    virtual Dual<double> eval(const std::vector<double>& vars) const = 0;
};

using NodePtr = std::shared_ptr<ExprNode>;

// ---------------- Variable Node ----------------
class VarNode : public ExprNode {
    int idx_;
    int nvars_;
public:
    VarNode(int i, int n) : idx_(i), nvars_(n) {}

    Dual<double> eval(const std::vector<double>& vars) const override {
        return Dual<double>(vars[idx_], idx_, nvars_);
    }
};

// ---------------- Constant Node ----------------
class ConstNode : public ExprNode {
    double val_;
    int nvars_;
public:
    ConstNode(double v, int n) : val_(v), nvars_(n) {}

    Dual<double> eval(const std::vector<double>&) const override {
        return Dual<double>(val_, nvars_);
    }
};

// ---------------- Operation Nodes ----------------
class UnaryOpNode : public ExprNode {
    std::function<Dual<double>(const Dual<double>&)> op_;
    NodePtr child_;
public:
    UnaryOpNode(auto op, NodePtr c) : op_(op), child_(c) {}

    Dual<double> eval(const std::vector<double>& vars) const override {
        return op_(child_->eval(vars));
    }
};

class BinaryOpNode : public ExprNode {
    std::function<Dual<double>(const Dual<double>&, const Dual<double>&)> op_;
    NodePtr left_, right_;
public:
    BinaryOpNode(auto op, NodePtr l, NodePtr r)
        : op_(op), left_(l), right_(r) {}

    Dual<double> eval(const std::vector<double>& vars) const override {
        return op_(left_->eval(vars), right_->eval(vars));
    }
};

// ============================================================
// Factory helpers (NO global variables)
// ============================================================
NodePtr variable(int idx, int nvars) {
    return std::make_shared<VarNode>(idx, nvars);
}

NodePtr constant(double c, int nvars) {
    return std::make_shared<ConstNode>(c, nvars);
}

// Operators
NodePtr operator+(NodePtr a, NodePtr b) {
    return std::make_shared<BinaryOpNode>(
        [](auto x, auto y) { return x + y; }, a, b);
}

NodePtr operator*(NodePtr a, NodePtr b) {
    return std::make_shared<BinaryOpNode>(
        [](auto x, auto y) { return x * y; }, a, b);
}

// Math wrappers
NodePtr sin(NodePtr a) {
    return std::make_shared<UnaryOpNode>(
        [](auto x) { return ad::sin(x); }, a);
}

NodePtr pow(NodePtr a, double p) {
    return std::make_shared<UnaryOpNode>(
        [p](auto x) { return ad::pow(x, p); }, a);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::cout << std::fixed << std::setprecision(6);

    int nvars = 2;

    auto x = variable(0, nvars);
    auto y = variable(1, nvars);

    // f(x,y) = x^3 + sin(x*y) + 3
    auto expr = pow(x, 3) + sin(x * y) + constant(3.0, nvars);

    std::vector<std::pair<double,double>> pts = {
        {1,1}, {2,1}, {1,2}, {2,2}
    };

    std::cout << "f(x,y) = x^3 + sin(x*y) + 3 " << "\n";

    for (auto [x0, y0] : pts) {
        Dual<double> res = expr->eval({x0, y0});

        std::cout << "f(" << x0 << "," << y0 << ") = " << res.value() << "\n";
        std::cout << "df/dx = " << res.derivative()[0] << "\n";
        std::cout << "df/dy = " << res.derivative()[1] << "\n\n";
    }

    // Correct analytic check at (1,1)
    double x0 = 1, y0 = 1;
    double dx = 3*x0*x0 + y0 * std::cos(x0*y0);
    double dy = x0 * std::cos(x0*y0);

    std::cout << "Analytical at (1,1):\n";
    std::cout << "df/dx = " << dx << "\n";
    std::cout << "df/dy = " << dy << "\n";

    return 0;
}
