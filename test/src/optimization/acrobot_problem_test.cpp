// Copyright (c) Sleipnir contributors

#include <numbers>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

// This problem's tight torque and joint velocity limits force the solver
// through many feasibility restoration iterations before converging.
//
// Regression test for feasibility restoration discarding the constraint
// second derivatives ∇ₓₓ²yᵀcₑ(x) from the restoration problem's Lagrangian
// Hessian (Eigen's SparseMatrix::resize() clears the matrix's values, so
// conservativeResize() must be used instead). Without the constraint
// curvature, restoration fails to make progress on this problem and the
// solver returns FEASIBILITY_RESTORATION_FAILED.
TEMPLATE_TEST_CASE("Problem - Acrobot", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr T dt(0.05);   // s
  constexpr int N = 120;  // 6 second horizon

  constexpr T τ_max(8);   // N·m
  constexpr T ω_max(20);  // rad/s

  // Acrobot parameters (uniform rods)
  constexpr T m1(1);           // kg
  constexpr T m2(1);           // kg
  constexpr T l1(1);           // m
  constexpr T lc1(0.5);        // m
  constexpr T lc2(0.5);        // m
  constexpr T I1(1.0 / 12.0);  // kg·m²
  constexpr T I2(1.0 / 12.0);  // kg·m²
  constexpr T g(9.806);        // m/s²

  slp::Problem<T> problem;

  // q₁ is the shoulder angle measured from the downward vertical and q₂ is
  // the relative elbow angle
  auto q1 = problem.decision_variable(1, N + 1);
  auto q2 = problem.decision_variable(1, N + 1);
  auto ω1 = problem.decision_variable(1, N + 1);
  auto ω2 = problem.decision_variable(1, N + 1);

  // Elbow torque (the shoulder is unactuated)
  auto τ = problem.decision_variable(1, N + 1);

  // Initial guess: linear interpolation from hanging to upright
  for (int k = 0; k < N + 1; ++k) {
    q1[0, k].set_value(T(std::numbers::pi) * T(k) / T(N));
  }

  // Joint accelerations from the manipulator equation
  //
  //   M(q)q̈ + h(q, q̇) + φ(q) = [0]
  //                             [τ]
  //
  // See equations (1) and (2) of Spong, "The swing up control problem for the
  // acrobot", 1995.
  auto accel = [&](int k) {
    using std::cos;
    using std::sin;

    auto s2 = sin(q2[0, k]);
    auto c2 = cos(q2[0, k]);

    auto d11 = m1 * lc1 * lc1 +
               m2 * (l1 * l1 + lc2 * lc2 + T(2) * l1 * lc2 * c2) + I1 + I2;
    auto d12 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + I2;
    auto d22 = m2 * lc2 * lc2 + I2;

    auto h1 = -m2 * l1 * lc2 * s2 * ω2[0, k] * ω2[0, k] -
              T(2) * m2 * l1 * lc2 * s2 * ω1[0, k] * ω2[0, k];
    auto h2 = m2 * l1 * lc2 * s2 * ω1[0, k] * ω1[0, k];

    auto φ1 = (m1 * lc1 + m2 * l1) * g * sin(q1[0, k]) +
              m2 * lc2 * g * sin(q1[0, k] + q2[0, k]);
    auto φ2 = m2 * lc2 * g * sin(q1[0, k] + q2[0, k]);

    auto a1 = -h1 - φ1;
    auto a2 = τ[0, k] - h2 - φ2;

    auto det = d11 * d22 - d12 * d12;
    return std::pair{(d22 * a1 - d12 * a2) / det, (d11 * a2 - d12 * a1) / det};
  };

  // Dynamics constraints - trapezoidal collocation
  for (int k = 0; k < N; ++k) {
    auto [α1_k, α2_k] = accel(k);
    auto [α1_k1, α2_k1] = accel(k + 1);

    problem.subject_to(q1[0, k + 1] ==
                       q1[0, k] + dt / T(2) * (ω1[0, k] + ω1[0, k + 1]));
    problem.subject_to(q2[0, k + 1] ==
                       q2[0, k] + dt / T(2) * (ω2[0, k] + ω2[0, k + 1]));
    problem.subject_to(ω1[0, k + 1] == ω1[0, k] + dt / T(2) * (α1_k + α1_k1));
    problem.subject_to(ω2[0, k + 1] == ω2[0, k] + dt / T(2) * (α2_k + α2_k1));
  }

  // Boundary conditions: hanging at rest → upright at rest
  problem.subject_to(q1[0, 0] == T(0));
  problem.subject_to(q2[0, 0] == T(0));
  problem.subject_to(ω1[0, 0] == T(0));
  problem.subject_to(ω2[0, 0] == T(0));
  problem.subject_to(q1[0, N] == T(std::numbers::pi));
  problem.subject_to(q2[0, N] == T(0));
  problem.subject_to(ω1[0, N] == T(0));
  problem.subject_to(ω2[0, N] == T(0));

  // Torque and joint velocity limits
  problem.subject_to(slp::bounds(-τ_max, τ, τ_max));
  problem.subject_to(slp::bounds(-ω_max, ω1, ω_max));
  problem.subject_to(slp::bounds(-ω_max, ω2, ω_max));

  // Minimize sum squared torques
  slp::Variable<T> J = T(0);
  for (int k = 0; k < N + 1; ++k) {
    J += τ[0, k] * τ[0, k];
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK_THAT(q1.value(0, 0), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(q2.value(0, 0), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(ω1.value(0, 0), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(ω2.value(0, 0), WithinAbs(T(0), T(1e-8)));

  // Verify solution
  for (int k = 0; k < N + 1; ++k) {
    // Torque limits
    CHECK(τ[0, k] >= -τ_max);
    CHECK(τ[0, k] <= τ_max);

    // Joint velocity limits
    CHECK(ω1[0, k] >= -ω_max);
    CHECK(ω1[0, k] <= ω_max);
    CHECK(ω2[0, k] >= -ω_max);
    CHECK(ω2[0, k] <= ω_max);
  }

  // Verify final state
  CHECK_THAT(q1.value(0, N), WithinAbs(T(std::numbers::pi), T(1e-8)));
  CHECK_THAT(q2.value(0, N), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(ω1.value(0, N), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(ω2.value(0, N), WithinAbs(T(0), T(1e-8)));
}
