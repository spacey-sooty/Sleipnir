// Copyright (c) Sleipnir contributors

// Determines a robot pose from the corner pixel locations of several AprilTags.
//
// The robot pose is constrained to be on the floor (z = 0).

#include <print>
#include <ranges>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

int main() {
  slp::Problem problem;

  // camera calibration
  constexpr double fx = 600;
  constexpr double fy = 600;
  constexpr double cx = 300;
  constexpr double cy = 150;

  // robot pose
  auto robot_x = problem.decision_variable();
  auto robot_y = problem.decision_variable();
  constexpr double robot_z = 0.0;
  auto robot_θ = problem.decision_variable();

  // cache autodiff variables
  auto sinθ = slp::sin(robot_θ);
  auto cosθ = slp::cos(robot_θ);

  slp::VariableMatrix field2robot{
      {cosθ, -sinθ, 0, robot_x},
      {sinθ, cosθ, 0, robot_y},
      {0, 0, 1, robot_z},
      {0, 0, 0, 1},
  };

  // robot is ENU, cameras are SDE
  constexpr Eigen::Matrix4d robot2camera{
      {0, 0, 1, 0},
      {-1, 0, 0, 0},
      {0, -1, 0, 0},
      {0, 0, 0, 1},
  };

  auto field2camera = field2robot * robot2camera;

  // list of points in field space to reproject. Each one is a 4x1 vector of
  // (x,y,z,1)
  std::vector field2points{slp::VariableMatrix{{2, 0 - 0.08255, 0.4, 1}}.T(),
                           slp::VariableMatrix{{2, 0 + 0.08255, 0.4, 1}}.T()};

  // List of points we saw the target at. These are exactly what we expect for a
  // camera located at 0,0,0 (hand-calculated)
  std::vector point_observations{std::pair{325, 30}, std::pair{275, 30}};

  for (size_t i = 0; i < point_observations.size(); i++) {
    point_observations[i].first = (point_observations[i].first - cx) / fx;
    point_observations[i].second = (point_observations[i].second - cy) / fy;
  }

  // initial guess at robot pose. We expect the robot to converge to 0,0,0
  robot_x.set_value(-0.1);
  robot_y.set_value(0.0);
  robot_θ.set_value(0.2);

  // field2camera * field2camera⁻¹ = I
  auto camera2field = slp::solve(field2camera, Eigen::Matrix4d::Identity());

  // Cost
  slp::Variable J = 0.0;
  for (const auto& [field2point, observation] :
       std::views::zip(field2points, point_observations)) {
    // camera2point = field2camera⁻¹ * field2point
    // field2camera * camera2point = field2point
    auto camera2point = camera2field * field2point;

    // point's coordinates in camera frame
    auto& x = camera2point[0];
    auto& y = camera2point[1];
    auto& z = camera2point[2];

    std::println("camera2point = {}, {}, {}", x.value(), y.value(), z.value());

    // coordinates observed at
    auto [xʼʼ_observed, yʼʼ_observed] = observation;

    auto xʼʼ = x / z;
    auto yʼʼ = y / z;

    std::println("Expected x {}, saw {}", xʼʼ.value(), xʼʼ_observed);
    std::println("Expected y {}, saw {}", yʼʼ.value(), yʼʼ_observed);

    auto xʼʼ_err = xʼʼ - xʼʼ_observed;
    auto yʼʼ_err = yʼʼ - yʼʼ_observed;

    // Cost function is square of reprojection error
    J += xʼʼ_err * xʼʼ_err + yʼʼ_err * yʼʼ_err;
  }

  problem.minimize(J);

  problem.solve({.diagnostics = true});

  std::println("x = {} m", robot_x.value());
  std::println("y = {} m", robot_y.value());
  std::println("θ = {} rad", robot_θ.value());
}
