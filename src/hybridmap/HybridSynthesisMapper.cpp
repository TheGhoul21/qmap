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

#include "hybridmap/HybridNeutralAtomMapper.hpp"
#include "hybridmap/NeutralAtomDefinitions.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace na {

std::vector<qc::fp> HybridSynthesisMapper::evaluateSynthesisSteps(
    qcs& synthesisSteps, const bool completeRemap, const bool alsoMap) {
  if (synthesisSteps.empty()) {
    return {};
  }
  if (!initialized) {
    initMapping(synthesisSteps[0].getNqubits());
    initialized = true;
  }
  std::vector<std::pair<qc::QuantumComputation, qc::fp>> candidates;
  size_t qcIndex = 0;
  for (auto& qc : synthesisSteps) {
    if (parameters.verbose) {
      spdlog::info("Evaluating synthesis step number {}", qcIndex);
    }
    const auto fidelity = evaluateSynthesisStep(qc, completeRemap);
    candidates.emplace_back(qc, fidelity);
    if (parameters.verbose) {
      spdlog::info("Fidelity: {}", fidelity);
    }
    ++qcIndex;
  }
  std::vector<qc::fp> fidelities;
  size_t bestIndex = 0;
  qc::fp bestFidelity = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    fidelities.push_back(candidates[i].second);
    if (candidates[i].second > bestFidelity) {
      bestFidelity = candidates[i].second;
      bestIndex = i;
    }
  }
  if (alsoMap && !candidates.empty()) {
    appendWithMapping(candidates[bestIndex].first, completeRemap);
  }
  return fidelities;
}

qc::fp
HybridSynthesisMapper::evaluateSynthesisStep(qc::QuantumComputation& qc,
                                             const bool completeRemap) const {
  NeutralAtomMapper tempMapper(arch, parameters);
  if (!completeRemap) {
    tempMapper.copyStateFrom(*this);
  }

  auto qcCopy = qc;
  qcCopy.reverse();
  // insert buffered operations
  for (const auto& opPointer : std::ranges::reverse_view(bufferedQc)) {
    qcCopy.emplace_back(opPointer->clone());
  }

  // Make a copy of qc to avoid modifying the original
  auto mappedQc = tempMapper.map(qcCopy, mapping);
  tempMapper.convertToAod();
  const auto results = tempMapper.schedule();
  return results.totalFidelities;
}

void HybridSynthesisMapper::appendWithMapping(qc::QuantumComputation& qc,
                                              const bool completeRemap) {
  if (mappedQc.empty() and bufferedQc.empty()) {
    initMapping(qc.getNqubits());
  }

  // add circuit to buffer
  for (const auto& op : qc) {
    this->bufferedQc.emplace_back(op->clone());
  }
  qc::QuantumComputation toBeMappedQc(qc.getNqubits());
  if (bufferedQc.size() > bufferSize) {
    // move beginning of buffer to synthesized circuit
    for (auto opPointer = bufferedQc.begin();
         opPointer < bufferedQc.end() - bufferSize; ++opPointer) {
      synthesizedQc.emplace_back((*opPointer)->clone());
      toBeMappedQc.emplace_back((*opPointer)->clone());
    }
    // remove the operations from the buffer
    bufferedQc.erase(bufferedQc.begin(),
                     bufferedQc.end() - static_cast<int64_t>(bufferSize));
  }

  if (completeRemap) {
    this->completeRemap(false);
  } else {
    mapAppend(toBeMappedQc, this->mapping);
  }
}

AdjacencyMatrix HybridSynthesisMapper::getCircuitAdjacencyMatrix() const {
  if (!initialized) {
    throw std::runtime_error(
        "Not yet initialized. Cannot get circuit adjacency matrix.");
  }
  const auto numCircQubits = synthesizedQc.getNqubits();
  AdjacencyMatrix adjMatrix(numCircQubits);

  for (uint32_t i = 0; i < numCircQubits; ++i) {
    for (uint32_t j = 0; j < numCircQubits; ++j) {
      if (i == j) {
        adjMatrix(i, j) = 0;
        continue;
      }
      const auto mappedI = mapping.getHwQubit(i);
      if (const auto mappedJ = mapping.getHwQubit(j);
          arch->getSwapDistance(mappedI, mappedJ) == 0) {
        adjMatrix(i, j) = 1;
        adjMatrix(j, i) = 1;
      } else {
        adjMatrix(i, j) = 0;
        adjMatrix(j, i) = 0;
      }
    }
  }
  return adjMatrix;
}

} // namespace na
