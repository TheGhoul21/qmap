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

#pragma once

#include "HybridNeutralAtomMapper.hpp"
#include "NeutralAtomArchitecture.hpp"
#include "NeutralAtomUtils.hpp"
#include "hybridmap/NeutralAtomDefinitions.hpp"
#include "hybridmap/NeutralAtomScheduler.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace na {

/**
 * @brief Bridges circuit synthesis (e.g., ZX extraction) with neutral-atom
 * mapping.
 * @details Derived from NeutralAtomMapper, this class maintains device state
 * and current mapping while accepting proposed synthesis steps. It evaluates
 * steps by mapping effort (e.g., required swaps/shuttling and timing), can
 * append steps with or without mapping, and exposes utilities to exchange
 * information with the synthesis algorithm.
 */
class HybridSynthesisMapper : public NeutralAtomMapper {
  using qcs = std::vector<qc::QuantumComputation>;

  qc::QuantumComputation synthesizedQc;
  uint32_t bufferSize;
  qc::QuantumComputation bufferedQc{};
  Mapping originalMapping;
  bool initialized = false;

  /**
   * @brief Evaluate a single proposed synthesis step.
   * @details Effort considers swaps/shuttling and execution time estimated by
   * the mapper.
   * @param qc Proposed synthesis subcircuit.
   * @param completeRemap
   * @return Scalar cost/effort score for mapping qc.
   */
  qc::fp evaluateSynthesisStep(qc::QuantumComputation& qc,
                               bool completeRemap = false) const;

public:
  // Constructors
  HybridSynthesisMapper() = delete;
  /**
   * @brief Construct with device and optional mapper parameters.
   * @param arch Neutral atom architecture.
   * @param params Optional mapper configuration parameters.
   */
  explicit HybridSynthesisMapper(
      const NeutralAtomArchitecture& arch,
      const MapperParameters& params = MapperParameters(),
      const uint32_t bufferSize = 0)
      : NeutralAtomMapper(arch, params), bufferSize(bufferSize) {}

  // Functions

  /**
   * @brief Initialize synthesized and mapped circuits and mapping structures.
   * @param nQubits Number of logical qubits to synthesize.
   */
  void initMapping(const size_t nQubits) {
    if (nQubits > arch->getNpositions()) {
      throw std::runtime_error("Not enough qubits in architecture.");
    }
    mappedQc = qc::QuantumComputation(arch->getNpositions());
    synthesizedQc = qc::QuantumComputation(nQubits);
    bufferedQc = qc::QuantumComputation(nQubits);
    mapping = Mapping(nQubits);
    initialized = true;
    originalMapping = mapping;
  }

  /**
   * @brief Complete a (re-)mapping of the synthesized circuit to hardware.
   */
  void completeRemap(const bool includeBuffer = true) {
    if (includeBuffer) {
      auto copyQC = getSynthesizedQc();
      map(copyQC, originalMapping);
    } else {
      map(synthesizedQc, originalMapping);
    }
  }

  /**
   * @brief Schedule the mapped circuit, appending buffered operations first.
   */
  [[nodiscard]] SchedulerResults
  schedule(const bool verboseArg = false, const bool createAnimationCsv = false,
           const qc::fp shuttlingSpeedFactor = 1.0) {
    mapAppend(bufferedQc, this->mapping);
    for (const auto& op : bufferedQc) {
      synthesizedQc.emplace_back(op->clone());
    }
    bufferedQc.clear();
    return NeutralAtomMapper::schedule(verboseArg, createAnimationCsv,
                                       shuttlingSpeedFactor);
  }

  /**
   * @brief Get the currently synthesized (unmapped) circuit.
   * @return Synthesized QuantumComputation.
   */
  [[nodiscard]] qc::QuantumComputation getSynthesizedQc() const {
    qc::QuantumComputation qc(synthesizedQc.getNqubits());
    qc.reserve(synthesizedQc.size() + bufferedQc.size());
    for (const auto& op : synthesizedQc) {
      qc.emplace_back(op->clone());
    }
    for (const auto& op : bufferedQc) {
      qc.emplace_back(op->clone());
    }
    return qc;
  }

  /**
   * @brief Export synthesized circuit as OpenQASM string.
   * @return QASM representation of the synthesized circuit.
   */
  [[nodiscard]] [[maybe_unused]] std::string getSynthesizedQcQASM() const {
    std::stringstream ss;
    const auto copyQC = getSynthesizedQc();
    copyQC.dumpOpenQASM(ss, false);
    return ss.str();
  }

  /**
   * @brief Save synthesized circuit as OpenQASM to a file.
   * @param filename Output filename.
   */
  [[maybe_unused]] void saveSynthesizedQc(const std::string& filename) const {
    std::ofstream ofs(filename);
    const auto copyQC = getSynthesizedQc();
    copyQC.dumpOpenQASM(ofs, false);
    ofs.close();
  }

  /**
   * @brief Evaluate candidate synthesis steps and optionally map the best.
   * @param synthesisSteps Vector of candidate subcircuits.
   * @param completeRemap If true, completely remap before evaluation.
   * @param alsoMap If true, append and map the best candidate.
   * @return List of fidelity scores for mapped steps (order matches input).
   */
  std::vector<qc::fp> evaluateSynthesisSteps(qcs& synthesisSteps,
                                             bool completeRemap = false,
                                             bool alsoMap = false);

  /**
   * @brief Append and map a subcircuit to hardware (may insert moves/SWAPs).
   * @param qc Subcircuit to append and map.
   * @param completeRemap If true, completely remap before appending.
   */
  void appendWithMapping(qc::QuantumComputation& qc,
                         bool completeRemap = false);

  /**
   * @brief Get the current device adjacency (connectivity) matrix.
   * @return Symmetric adjacency matrix for the neutral atom hardware.
   */
  [[nodiscard]] AdjacencyMatrix getCircuitAdjacencyMatrix() const;
};
} // namespace na
