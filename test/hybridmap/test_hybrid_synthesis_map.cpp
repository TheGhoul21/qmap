/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#include "hybridmap/HybridSynthesisMapper.hpp"
#include "hybridmap/NeutralAtomArchitecture.hpp"
#include "ir/QuantumComputation.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace na {

struct TestParams {
  std::string architecture;
  bool completeRemap;
  bool alsoMap;
};

class TestParametrizedHybridSynthesisMapper
    : public testing::TestWithParam<TestParams> {
protected:
  std::string testArchitecturePath = "architectures/";
  std::vector<qc::QuantumComputation> circuits;

  void SetUp() override {
    testArchitecturePath += GetParam().architecture + ".json";
    qc::QuantumComputation qc1(3);
    qc1.x(0);
    qc1.cx(0, 1);
    qc1.cx(1, 2);
    circuits.push_back(qc1);

    qc::QuantumComputation qc2(3);
    qc2.move(0, 2);
    qc2.x(0);
    circuits.push_back(qc2);
  }

  // Test the HybridSynthesisMapper class
};

TEST_P(TestParametrizedHybridSynthesisMapper, AdjacencyMatrix) {
  const auto arch = NeutralAtomArchitecture(testArchitecturePath);
  auto mapper = HybridSynthesisMapper(arch);
  mapper.initMapping(3);
  auto adjMatrix = mapper.getCircuitAdjacencyMatrix();
  EXPECT_EQ(adjMatrix.size(), 3);
  EXPECT_TRUE(adjMatrix(0, 2) == 0 || adjMatrix(0, 2) == 1);
}

TEST_P(TestParametrizedHybridSynthesisMapper, EvaluateSynthesisStep) {
  const auto arch = NeutralAtomArchitecture(testArchitecturePath);
  auto params = MapperParameters();
  params.verbose = true;
  auto mapper = HybridSynthesisMapper(arch, params);
  // Intentionally not initializing the mapper to test error handling
  EXPECT_THROW(static_cast<void>(mapper.getCircuitAdjacencyMatrix()),
               std::runtime_error);
  // Initializing with too many qubits to test error handling
  EXPECT_THROW(mapper.initMapping(50), std::runtime_error);
  const auto best = mapper.evaluateSynthesisSteps(circuits, true);
  EXPECT_EQ(best.size(), 2);
  EXPECT_GE(best[0], 0);
  EXPECT_GE(best[1], 0);
}

INSTANTIATE_TEST_SUITE_P(
    HybridSynthesisMapperTestSuite, TestParametrizedHybridSynthesisMapper,
    ::testing::Values(TestParams{"rubidium_gate", false, false},
                      TestParams{"rubidium_gate", true, false},
                      TestParams{"rubidium_gate", false, true},
                      TestParams{"rubidium_gate", true, true},
                      TestParams{"rubidium_hybrid", false, false},
                      TestParams{"rubidium_hybrid", true, false},
                      TestParams{"rubidium_hybrid", false, true},
                      TestParams{"rubidium_hybrid", true, true},
                      TestParams{"rubidium_shuttling", false, false},
                      TestParams{"rubidium_shuttling", true, false},
                      TestParams{"rubidium_shuttling", false, true},
                      TestParams{"rubidium_shuttling", true, true}));

class TestHybridSynthesisMapper : public testing::Test {
protected:
  NeutralAtomArchitecture arch =
      NeutralAtomArchitecture("architectures/rubidium_gate.json");
  HybridSynthesisMapper mapper = HybridSynthesisMapper(arch);
  qc::QuantumComputation qc;

  void SetUp() override {
    qc = qc::QuantumComputation(3);
    qc.x(0);
    qc.cx(0, 1);
    qc.cx(1, 2);

    mapper.initMapping(3);
  }
};

TEST_F(TestHybridSynthesisMapper, MapAppend) {
  mapper.appendWithMapping(qc);
  const auto synthesizedQc = mapper.getSynthesizedQc();
  EXPECT_EQ(synthesizedQc.getNqubits(), 3);
  EXPECT_GE(synthesizedQc.getNops(), 3);
}

TEST_F(TestHybridSynthesisMapper, Output) {
  mapper.appendWithMapping(qc, true);
  const auto qasm = mapper.getSynthesizedQcQASM();
  EXPECT_FALSE(qasm.empty());
  const auto tempDir = std::filesystem::temp_directory_path();
  const auto qasmPath = tempDir / "test_output.qasm";
  mapper.saveSynthesizedQc(qasmPath.string());
  EXPECT_TRUE(std::filesystem::exists(qasmPath));
  EXPECT_GT(std::filesystem::file_size(qasmPath), 0);
  std::filesystem::remove(qasmPath);
}

} // namespace na
